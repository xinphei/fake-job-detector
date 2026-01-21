
import os, json, re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import hstack, csr_matrix
from io import BytesIO
import pycountry
import matplotlib.pyplot as plt


# fix for reportlab error
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    st.warning("ReportLab not found. PDF generation will be disabled.")

st.set_page_config(page_title="Fake Job Posting Detector (XGB + LIME)", layout="wide")
USD_TO_MYR = 4.06

# LIME safe import
try:
    from lime.lime_text import LimeTextExplainer
    LIME_OK = True
except Exception:
    LIME_OK = False

DEPLOY_DIR = "artifacts_deploy"
bundle_path = os.path.join(DEPLOY_DIR, "deploy_bundle.json")

if os.path.exists(bundle_path):
    with open(bundle_path, "r") as f:
        bundle = json.load(f)

    missing = [
        fname for fname in bundle.values()
        if not os.path.exists(os.path.join(DEPLOY_DIR, fname))
    ]

    if missing:
        st.error("Missing deployment files referenced in deploy_bundle.json")
        st.write(missing)
        st.stop()
else:
    required_files = [
        "best_model.joblib", "tfidf.joblib", "best_thresholds.json",
        "scaler.joblib", "ohe.joblib", "label_encoders.joblib",
        "structured_columns.joblib", "feature_groups.joblib"
    ]

    missing = [
        f for f in required_files
        if not os.path.exists(os.path.join(DEPLOY_DIR, f))
    ]

    if missing:
        st.error("Missing deployment files (no deploy_bundle.json found)")
        st.write(missing)
        st.stop()

# Ensure NLTK resources
import nltk
from bs4 import BeautifulSoup
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

@st.cache_resource
def ensure_nltk():
    import nltk

    # (nltk_path_to_check, download_package_name)
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"), 
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]

    for path, pkg in resources:
        try:
            nltk.data.find(path)       # ✅ check if already exists
        except LookupError:
            nltk.download(pkg, quiet=True)

    return True

ensure_nltk()
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith("J"): return "a"
    if tag.startswith("V"): return "v"
    if tag.startswith("N"): return "n"
    if tag.startswith("R"): return "r"
    return "n"

@st.cache_data(show_spinner=False)
def read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(file_bytes))

    
# Helper functions for preprocessing
def extract_location_features(location):
    if pd.isna(location) or str(location).strip() == "":
        return "unknown", "unknown", "unknown"

    parts = [p.strip().lower() for p in str(location).split(",") if p.strip()]

    # training format: "country, state, city"
    if len(parts) >= 3:
        country_raw, state_raw, city_raw = parts[0], parts[1], parts[2]
    elif len(parts) == 2:
        country_raw, state_raw, city_raw = parts[0], parts[1], "unknown"
    elif len(parts) == 1:
        country_raw, state_raw, city_raw = parts[0], "unknown", "unknown"
    else:
        return "unknown", "unknown", "unknown"

    # normalize country to alpha-2 if user typed full name 
    country = country_raw
    if not (len(country_raw) in (2, 3) and country_raw.replace("-", "").isalnum()):
        try:
            cobj = pycountry.countries.search_fuzzy(country_raw)[0]
            country = cobj.alpha_2.lower()
        except Exception:
            country = country_raw

    # normalize state to code ONLY for US 
    state = state_raw
    if country == "us":
        # if already 2-letter code like "ny", keep it
        if not (len(state_raw) == 2 and state_raw.isalpha()):
            # try match full state name -> US-XX
            try:
                for sub in pycountry.subdivisions:
                    if sub.country_code == "US" and sub.name.lower() == state_raw:
                        state = sub.code.split("-")[-1].lower()  # "US-NY" -> "ny"
                        break
            except Exception:
                pass

    return city_raw, state, country


def extract_salary_features(salary_range):
    if pd.isna(salary_range) or str(salary_range).strip() == "":
        return 0.0, 0.0, "myr", 0
    s = str(salary_range).lower().strip()
    is_usd = ("$" in s) or ("usd" in s) or ("us$" in s)
    
    nums = re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", s)
    if len(nums) == 0:
        return 0.0, 0.0, "myr", 0
        
    vals = [float(x.replace(",", "")) for x in nums]
    
    if len(vals) == 1:
        smin = smax = vals[0]
    else:
        smin, smax = vals[0], vals[1]
        if smin > smax: smin, smax = smax, smin
        
    # convery USD input to RM (training is RM)   
    if is_usd:
        smin *= USD_TO_MYR
        smax *= USD_TO_MYR
        
    return float(smin), float(smax), "myr", 1

def build_combined_text(row):
    fields = ["title", "location", "department", "company_profile", "description", "requirements", "benefits"]
    parts = []
    for f in fields:
        val = row.get(f, "")
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    return ". ".join(parts)

def clean_text_for_ml(text):
    if not isinstance(text, str) or not text.strip(): return ""
    text = contractions.fix(text)
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    # remove dataset URL placeholders like "URL_d9efc..." or "#URL_d9efc..."
    text = re.sub(r'#?url_[0-9a-f]+', ' ', text, flags=re.IGNORECASE)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return ""
        
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    if not tokens:
        return ""
        
    pos_tags = nltk.pos_tag(tokens)
    lemmas = [lemmatizer.lemmatize(w, get_wordnet_pos(p)) for w, p in pos_tags]
    return " ".join(lemmas)

def preprocess_one_row(raw_df: pd.DataFrame) -> pd.DataFrame:
    dfp = raw_df.copy().reset_index(drop=True)
    current_idx = 0
    
    # fill required raw cols
    text_cols = ["title","location","department","salary_range","company_profile","description","requirements","benefits"]
    for c in text_cols:
        if c not in dfp.columns: dfp[c] = ""
        dfp[c] = dfp[c].fillna("").astype(str)

    cat_cols = ["employment_type","required_experience","required_education","industry","function"]
    for c in cat_cols:
        if c not in dfp.columns: dfp[c] = "Unknown"
        dfp[c] = dfp[c].fillna("Unknown").astype(str)

    for b in ["telecommuting","has_company_logo","has_questions"]:
        if b not in dfp.columns: dfp[b] = 0
        dfp[b] = pd.to_numeric(dfp[b], errors="coerce").fillna(0).astype(int)

    # Reset index to 0 so we can always use .loc[0] regardless of CSV row number
    dfp = dfp.reset_index(drop=True)
    current_idx = 0
    
    # location engineered
    city, state, country = extract_location_features(dfp.loc[current_idx, "location"])
    dfp["city"] = city
    dfp["state"] = state
    dfp["country"] = country

    # salary engineered - Use current_idx
    sr = dfp.loc[current_idx, "salary_range"]

    smin, smax, cur, has_salary = extract_salary_features(sr)

    dfp["has_salary"] = int(has_salary)
    dfp["salary_min"] = float(smin)
    dfp["salary_max"] = float(smax)
    dfp["salary_avg"] = (dfp["salary_min"] + dfp["salary_max"]) / 2.0
    dfp["salary_span"] = np.maximum(dfp["salary_max"] - dfp["salary_min"], 0.0)
    
    dfp["currency"] = cur  # always "myr" now
    
    dfp["log_salary_avg"] = np.log1p(dfp["salary_avg"])
    dfp["log_salary_span"] = np.log1p(dfp["salary_span"])

    # text + ML clean_text (used by TF-IDF)
    dfp["text"] = dfp.apply(build_combined_text, axis=1)
    # FALLBACK: If user enters nothing, provide a placeholder so TF-IDF doesn't crash
    if dfp["text"].iloc[0].strip() == "":
        dfp["text"] = "Unknown Job Title"
        
    dfp["clean_text"] = dfp["text"].apply(clean_text_for_ml)

    # length + flags
    dfp["text_length"] = dfp["text"].str.len()
    dfp["word_count"] = dfp["clean_text"].str.split().str.len().fillna(0).astype(int)
    dfp["log_word_count"] = np.log1p(dfp["word_count"])

    dfp["has_company_profile"] = (dfp["company_profile"].str.strip() != "").astype(int)
    dfp["has_requirements"] = (dfp["requirements"].str.strip() != "").astype(int)
    dfp["has_benefits"] = (dfp["benefits"].str.strip() != "").astype(int)
    dfp["title_length"] = dfp["title"].str.len()

    return dfp

# load artifacts
@st.cache_resource
def load_artifacts():
    # Use original bundle logic to ensure it finds your specific files
    if os.path.exists(bundle_path):
        with open(bundle_path, "r") as f:
            b = json.load(f)

        required_keys = [
            "model","tfidf","scaler","ohe","label_encoders",
            "structured_columns","feature_groups","thresholds"
        ]
        for k in required_keys:
            if k not in b:
                raise KeyError(f"deploy_bundle.json missing key: {k}")

        def fp(key):
            return os.path.join(DEPLOY_DIR, b[key])

        model = joblib.load(fp("model"))
        tfidf = joblib.load(fp("tfidf"))
        scaler = joblib.load(fp("scaler"))
        ohe = joblib.load(fp("ohe"))
        label_encoders = joblib.load(fp("label_encoders"))
        structured_cols = joblib.load(fp("structured_columns"))
        feature_groups = joblib.load(fp("feature_groups"))
        with open(fp("thresholds"), "r") as f:
            thr = json.load(f)
    else:
        model = joblib.load(os.path.join(DEPLOY_DIR, "best_model.joblib"))
        tfidf = joblib.load(os.path.join(DEPLOY_DIR, "tfidf.joblib"))
        scaler = joblib.load(os.path.join(DEPLOY_DIR, "scaler.joblib"))
        ohe = joblib.load(os.path.join(DEPLOY_DIR, "ohe.joblib"))
        label_encoders = joblib.load(os.path.join(DEPLOY_DIR, "label_encoders.joblib"))
        structured_cols = joblib.load(os.path.join(DEPLOY_DIR, "structured_columns.joblib"))
        feature_groups = joblib.load(os.path.join(DEPLOY_DIR, "feature_groups.joblib"))
        with open(os.path.join(DEPLOY_DIR, "best_thresholds.json"), "r") as f:
            thr = json.load(f)

    best_t_xgb = float(thr.get("best_t_xgb", 0.5))
    return model, tfidf, scaler, ohe, label_encoders, structured_cols, feature_groups, best_t_xgb

model, tfidf, scaler, ohe, label_encoders, structured_cols, feature_groups, best_t_xgb = load_artifacts()

SEED = 42


# model.classes_ is the true order used in predict_proba columns
MODEL_CLASSES = list(model.classes_)          # e.g. [0,1] or [1,0]
IDX_REAL = MODEL_CLASSES.index(0)             # class 0 = Real
IDX_FAKE = MODEL_CLASSES.index(1)             # class 1 = Fake
CLASS_NAMES = ["Real" if c == 0 else "Fake" for c in MODEL_CLASSES]

explainer = LimeTextExplainer(class_names=CLASS_NAMES, random_state=SEED) if LIME_OK else None


def get_risk_badge(proba_fake: float, threshold: float):
    # Simple rule:
    # - HIGH: >= threshold
    # - MEDIUM: >= half of threshold
    # - LOW: below half threshold
    if proba_fake >= threshold:
        return "HIGH", "#c0392b", "🚨"
    elif proba_fake >= 0.5 * threshold:
        return "MEDIUM", "#f39c12", "⚠️"
    else:
        return "LOW", "#27ae60", "✅"


def safe_label_encode(col, val):
    le = label_encoders.get(col)
    if le is None:
        return 0
    v = str(val)
    if v in le.classes_:
        return int(le.transform([v])[0])
    return 0

def build_structured_from_dfproc(df_proc: pd.DataFrame) -> np.ndarray:
    dfp = df_proc.copy()

    fg = feature_groups
    numeric_cols = fg["numeric_cols"]
    low_card_cols = fg["low_card_cols"]
    high_card_cols = fg["high_card_cols"]
    binary_cols = fg["binary_cols"]

    base = {}

    for c in binary_cols:
        base[c] = int(dfp[c].iloc[0]) if c in dfp.columns else 0

    for c in numeric_cols:
        base[c] = float(dfp[c].iloc[0]) if c in dfp.columns else 0.0

    for c in high_card_cols:
        base[c] = safe_label_encode(c, dfp[c].iloc[0] if c in dfp.columns else "unknown")

    base_df = pd.DataFrame([base])

    for c in numeric_cols:
        if c not in base_df.columns:
            base_df[c] = 0.0
    base_df[numeric_cols] = scaler.transform(base_df[numeric_cols].astype(float))

    low_input = {}
    for c in low_card_cols:
        low_input[c] = str(dfp[c].iloc[0]) if c in dfp.columns else "Unknown"
    low_df = pd.DataFrame([low_input])

    ohe_arr = ohe.transform(low_df)
    if hasattr(ohe_arr, "toarray"):
        ohe_arr = ohe_arr.toarray()
    ohe_df = pd.DataFrame(ohe_arr, columns=ohe.get_feature_names_out(low_card_cols))

    full_df = pd.concat([base_df.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)

    for c in structured_cols:
        if c not in full_df.columns:
            full_df[c] = 0.0
    full_df = full_df[structured_cols]

    return full_df.values.astype(float)

def predict_proba_from_raw_df(raw_df: pd.DataFrame):
    df_proc = preprocess_one_row(raw_df)
    if "clean_text" not in df_proc.columns:
        raise ValueError("Missing clean_text")
    if not df_proc["clean_text"].iloc[0].strip():
        raise ValueError("Empty text")

    clean_text = df_proc["clean_text"].values
    struct_vec = build_structured_from_dfproc(df_proc)

    X_text = tfidf.transform(clean_text)
    X_struct_sp = csr_matrix(struct_vec)
    X_all = hstack([X_text, X_struct_sp]).tocsr()

    proba = model.predict_proba(X_all)
    return proba, df_proc, struct_vec

def predict_proba_for_lime(text_list, struct_row_vector_1d):
    cleaned_list = [clean_text_for_ml(t) for t in text_list]
    X_text_new = tfidf.transform(cleaned_list)

    X_struct_rep = np.repeat(
        struct_row_vector_1d.reshape(1, -1),
        repeats=len(text_list),
        axis=0
    )
    X_struct_sp = csr_matrix(X_struct_rep)

    X_new = hstack([X_text_new, X_struct_sp]).tocsr()
    return model.predict_proba(X_new)  # (n,2)

# PDF report generation
def generate_pdf(inputs, proba_fake, pred, lime_features, lime_label_idx):

    if not HAS_REPORTLAB:
        return None

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    text = c.beginText(40, height - 50)
    
    text.setFont("Helvetica-Bold", 14)
    text.textLine("Fake Job Posting Detection Report")
    text.textLine("=" * 50)
    
    text.moveCursor(0, 15)
    result_text = "POTENTIALLY FAKE" if pred == 1 else "LIKELY REAL"
    text.setFont("Helvetica-Bold", 12)
    text.textLine(f"FINAL DECISION: {result_text}")
    text.setFont("Helvetica", 10)
    text.textLine(f"Suspicion Score: {proba_fake:.2%}")
    text.textLine(f"Risk Alert Level: {best_t_xgb*100:.0f}%")
    text.moveCursor(0, 20)
    text.setFont("Helvetica-Bold", 11)
    text.textLine("SUBMITTED JOB DETAILS:")
    text.setFont("Helvetica", 9)
    
    for k, v in inputs.items():
        clean_v = str(v).replace('\n', ' ').strip()
        display_v = (clean_v[:85] + '...') if len(clean_v) > 85 else clean_v
        text.textLine(f"{k.upper()}: {display_v}")
    text.moveCursor(0, 20)
    
  

    if lime_features:
        text.setFont("Helvetica-Bold", 11)
        text.textLine("TOP INFLUENTIAL KEYWORDS:")

        pos_supports = "Fake" if lime_label_idx == IDX_FAKE else "Real"
        neg_supports = "Real" if lime_label_idx == IDX_FAKE else "Fake"

        text.setFont("Helvetica", 9)
        for word, weight in lime_features[:10]:
            if weight >= 0:
                text.textLine(f"• {word}: supports {pos_supports} | Weight: {weight:.4f}")
            else:
                text.textLine(f"• {word}: supports {neg_supports} | Weight: {abs(weight):.4f}")

    c.drawText(text)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

def split_lime_support(exp, idx, top_k=8):
    lst = exp.as_list(label=idx)

    if idx == IDX_REAL:
        supports_real = [(w, wt) for w, wt in lst if wt > 0]
        supports_fake = [(w, -wt) for w, wt in lst if wt < 0]
    else:  # idx == IDX_FAKE
        supports_fake = [(w, wt) for w, wt in lst if wt > 0]
        supports_real = [(w, -wt) for w, wt in lst if wt < 0]

    supports_real = sorted(supports_real, key=lambda x: x[1], reverse=True)[:top_k]
    supports_fake = sorted(supports_fake, key=lambda x: x[1], reverse=True)[:top_k]
    return supports_real, supports_fake

    
# UI DESIGN SECTION
st.markdown("""
<style>
/* ---------- RISK BADGE ---------- */
.risk-badge{
  display:inline-flex;
  align-items:center;
  gap:8px;
  padding:6px 12px;
  border-radius:999px;
  font-size:14px;
  font-weight:800;
  color:white;
  white-space:nowrap;
}



.stApp {
    background-color: #b4cefa; 
}

/* Hide Streamlit default header */
header[data-testid="stHeader"] {
    display: none;
}

/* Remove extra top padding left by header */
div.block-container{
  max-width: 1150px;     /* try 1050–1250 */
  padding-left: 3rem;
  padding-right: 3rem;
  padding-top: 2rem;
}

/* GLOBAL BASE */
html, body, [class*="css"]  {
    font-size: 18px !important;
}

/* HEADINGS */
h1 {
    font-size: 56px !important;
    font-weight: 900 !important;
    letter-spacing: -0.5px;
    margin-bottom: 10px;
    color: #1f2937 !important;  /* dark slate */
}

h2 {
    font-size: 26px !important;
    font-weight: 700 !important;
}
h3 {
    font-size: 22px !important;
    font-weight: 700 !important;
}

/* LABELS (Title, Location, etc.) */
/* Targeting the specific Streamlit Label Container */
div[data-testid="stWidgetLabel"] p {
    font-size: 24px !important;  /* Much larger */
    font-weight: 900 !important;  /* Maximum thickness */
    color: #301934 !important;    /* Your dark purple */
    line-height: 1.2 !important;
    margin-bottom: 5px !important;
}

/* Fallback for other label types */
label {
    font-size: 24px !important;
    font-weight: 900 !important;
    color: #301934 !important;
}

/* INPUT TEXT (Inside the white boxes) */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
div[data-baseweb="select"] > div {
    background-color: #f8fafc !important; /* Soft Off-White */
    border: 1px solid #e2e8f0 !important; /* Subtle border */
    color: #1f2937 !important;
}

/* PLACEHOLDER / HELP TEXT */
.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    font-size: 16px !important;
    color: #374151 !important;
}

/* Checkbox & Toggle LABEL text */
div[data-testid="stCheckbox"] label,
div[data-testid="stToggle"] label {
    font-size: 11px !important;
    font-weight: 300 !important;   /* normal */
    color: #1f2937 !important;
}

/* Remove extra bold span inside label */
div[data-testid="stCheckbox"] label span,
div[data-testid="stToggle"] label span {
    font-weight: 300 !important;
}

/* Reduce vertical spacing */
div[data-testid="stCheckbox"],
div[data-testid="stToggle"] {
    margin-top: 4px;
    margin-bottom: 4px;
}

/* TABS */
.stTabs [data-baseweb="tab"] {
    font-size: 18px !important;
    font-weight: 700 !important;
    padding: 10px 18px !important;
    border-radius: 12px 12px 0 0 !important;
    color: #002366 !important;
}

/* Active tab */
.stTabs [aria-selected="true"] {
    background-color: #002366 !important;
    color: white !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #6A5ACD !important; /* Slate blue hover */
    background-color: rgba(255, 255, 255, 0.8) !important;
}

/* Remove underline / red line */
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

/* BUTTONS */
div.stButton > button {
    font-size: 18px !important;
    font-weight: 800 !important;
    padding: 14px 24px !important;
    border-radius: 10px !important;
    background-color: #002366 !important;
    color: white !important;
    border: none !important;
}
div.stButton > button:hover {
    background-color: #003399 !important;
}

/* DOWNLOAD (PDF) BUTTON */
div.stDownloadButton > button {
    font-size: 18px !important;
    font-weight: 800 !important;
    padding: 14px 20px !important;
    border-radius: 10px !important;
    background-color: #b00020 !important;
    color: white !important;
    border: none !important;
}

/* CARDS */
.card {
    background-color: white;
    padding: 22px;
    border-radius: 14px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04) !important;
    margin-bottom: 20px;
}

/* REMOVE HEADING LINK ICON */
a[title="Copy link to this heading"] {
    display: none !important;
}

/* 🔥 REMOVE the form border */
div[data-testid="stForm"] {
  border: none !important;
  padding: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}

/* 🔵 Style the Predict button (form submit) */
div[data-testid="stFormSubmitButton"] > button {
  font-size: 18px !important;
  font-weight: 800 !important;
  padding: 14px 24px !important;
  border-radius: 10px !important;
  background-color: #002366 !important;  /* dark blue */
  color: white !important;
  border: none !important;
}

/* Hover effect */
div[data-testid="stFormSubmitButton"] > button:hover {
  background-color: #003399 !important;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<h1>🕵️ Fake Job Posting Detector</h1>
<h3>AI-powered job scam detection</h3>
<p style="color:#555;">Assess whether a posting is <b>Real</b> or <b>Potentially Fake</b>.</p>
<p style="color:#cf2121; font-style: italic;">The more information you provide, the more precise our analysis will be.</p>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Single Job Check", "Bulk CSV Detection"])

with tab1:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        title = st.text_input("Title")
        location = st.text_input("Location", help="Format: country, state, city (e.g., US, NY, New York)")
        department = st.text_input("Department", "Unknown")
        salary_range = st.text_input("Salary Range", help="Default is MYR (e.g., 5000-7000). Use $ or USD if the amount is in USD.")
        employment_type = st.text_input("Employment Type", "Unknown")
        required_experience = st.text_input("Required Experience", "Unknown")
        required_education = st.text_input("Required Education", "Unknown")
        industry = st.text_input("Industry", "Unknown")
        function_ = st.text_input("Function", "Unknown")
        
    with c2:
        company_profile = st.text_area("Company Profile", height=110)
        description = st.text_area("Description", height=160)
        requirements = st.text_area("Requirements", height=130)
        benefits = st.text_area("Benefits", height=110)

        telecommuting = st.toggle("🌍 Remote / Telecommuting", value=False)
        has_company_logo = st.toggle("🏢 Company Logo Provided", value=False)
        has_questions = st.toggle("❓ Screening Questions Included", value=False)

    show_lime = st.checkbox("Show Explanation", value=True, key="show_lime_single")
    with st.form("predict_form"):
        submitted = st.form_submit_button("Predict")

    if submitted:
        inputs = {
            "title": title, "location": location, "department": department, "salary_range": salary_range,
            "employment_type": employment_type, "required_experience": required_experience,
            "required_education": required_education, "industry": industry, "function": function_,
            "company_profile": company_profile, "description": description, "requirements": requirements,
            "benefits": benefits, "telecommuting": int(telecommuting), "has_company_logo": int(has_company_logo),
            "has_questions": int(has_questions), "fraudulent": 0
        }

        # block empty submissions
        min_text = (title + " " + description + " " + requirements + " " + company_profile).strip()
        if min_text == "":
            st.error("Please enter at least a Title or Description before predicting.")
            st.stop()


        try:
            proba, df_proc, struct_vec = predict_proba_from_raw_df(pd.DataFrame([inputs]))
            
            # SAVE TO SESSION STATE: "freezes" the result so it stays on screen
            st.session_state['result'] = {
                'proba_fake': float(proba[0][IDX_FAKE]),
                'df_proc': df_proc,
                'struct_vec': struct_vec,
                'inputs': inputs
            }
        except Exception as e:
            st.error(f"Prediction Error: {e}")
    
        # DISPLAY RESULTS (If they exist in Session State)
    if "result" in st.session_state:
        res = st.session_state["result"]
        proba_fake = res["proba_fake"]
        pred = 1 if proba_fake >= best_t_xgb else 0

        # UI: Cards and Charts
        res_col1, res_col2 = st.columns([1.2, 1])

        with res_col1:
            result_color = "#c0392b" if pred == 1 else "#27ae60"
            risk_label, risk_color, risk_icon = get_risk_badge(proba_fake, best_t_xgb)
            headline = "🚨 Potentially FAKE" if pred == 1 else "✅ Likely REAL"

            # Use direct variables {headline} etc. inside f-string
            # MOVE TAGS TO THE FAR LEFT MARGIN
            html = f"""
            <div style="height: 100%; display: flex;">
            <div class="card" style="border-left:10px solid {result_color}; margin: 0; flex: 1; display: flex; flex-direction: column; justify-content: center; min-height: 435px;">
            <div style="display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom: 20px;">
            <h2 style="margin:0; font-size: 30px;">{headline}</h2>
            <span class="risk-badge" style="background:{risk_color}; padding: 8px 16px;">
            {risk_icon} {risk_label} RISK
            </span>
            </div>
            <p style="font-size: 24px; margin-bottom: 10px;"><b>Suspicion Score:</b> {proba_fake:.2%}</p>
            <p style="font-size: 24px;"><b>Threshold:</b> {best_t_xgb*100:.0f}%</p>
            </div>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)

        with res_col2:
            
            
            # 1. Setup the Plot
            fig, ax = plt.subplots(figsize=(4, 4))
            fig.patch.set_facecolor('#b4cefa') # App background color
            ax.set_facecolor('#b4cefa')

            # 2. DRAW the Pie Chart 
            wedges, texts, autotexts = ax.pie(
                [1 - proba_fake, proba_fake], 
                labels=["Real", "Fake"], 
                autopct="%.1f%%", 
                colors=["#27ae60", "#c0392b"], 
                startangle=140, 
                pctdistance=0.75,
                textprops={'fontsize': 16}
            )
        
            # 3. Style the labels
            texts[0].set_color("#27ae60")
            texts[0].set_fontsize(20)
            texts[0].set_weight("bold")
            texts[1].set_color("#c0392b")
            texts[1].set_fontsize(20)
            texts[1].set_weight("bold")
            
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_weight('bold')
        
            ax.axis('equal') 
            plt.tight_layout()

            # 4. display the finished plot to Streamlit
            st.pyplot(fig, clear_figure=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # LIME SECTION
        lime_feat_list = []
        supports_real, supports_fake = [], []

        if show_lime and LIME_OK:
            try:
                st.markdown("""
                <div class="card" style="margin-top:10px;">
                <h3 style="margin:0;">🔍 Why did the model make this decision?</h3>
                <p style="margin:6px 0 0 0; color:#334155;">
                    <b>Note:</b> LIME weights are not probabilities. They show which words push the model locally.
                    Even if some words look scam-like, the final decision is based on the overall balance + structured features.
                </p>
                </div>
                """, unsafe_allow_html=True)

                sample_text_raw = res["df_proc"]["text"].iloc[0]
                struct_vec_1d = res["struct_vec"].ravel()

                # predicted class INDEX (NOT 0/1 label)
                pred_idx = IDX_FAKE if proba_fake >= best_t_xgb else IDX_REAL
                st.session_state["result"]["lime_label_idx"] = pred_idx

                exp = explainer.explain_instance(
                    str(sample_text_raw),
                    classifier_fn=lambda txts: predict_proba_for_lime(txts, struct_vec_1d),
                    num_features=12,
                    labels=[IDX_REAL, IDX_FAKE],       
                    num_samples=1000
                )

                # st.write("MODEL_CLASSES:", MODEL_CLASSES)
                # st.write("pred_idx used for LIME:", pred_idx, "=>", CLASS_NAMES[pred_idx])
                # st.write("Top LIME list:", exp.as_list(label=pred_idx)[:5])


                # Use ONLY the predicted label explanation
                lime_feat_list = exp.as_list(label=pred_idx)
                # st.write("DEBUG (signed LIME weights for predicted label):", lime_feat_list)
                st.session_state["result"]["lime_feat_list"] = lime_feat_list

                # Split supports using sign from SAME label
                pos = [(w, wt) for w, wt in lime_feat_list if wt > 0]
                neg = [(w, -wt) for w, wt in lime_feat_list if wt < 0]

                pos = sorted(pos, key=lambda x: x[1], reverse=True)[:8]
                neg = sorted(neg, key=lambda x: x[1], reverse=True)[:8]

                if pred_idx == IDX_REAL:
                    supports_real = pos
                    supports_fake = neg
                else:
                    supports_fake = pos
                    supports_real = neg

                colR, colF = st.columns([1, 1], gap="large")
                with colR:
                    st.markdown("#### 🛡️ Words supporting REAL")
                    if supports_real:
                        items = "".join([f"<li><b>{w}</b> ({s:.3f})</li>" for w, s in supports_real])
                        st.markdown(f"<ul style='padding-left:26px;'>{items}</ul>", unsafe_allow_html=True)
                    else:
                        st.write("No strong indicators found.")

                with colF:
                    st.markdown("#### 🚨 Words supporting FAKE")
                    if supports_fake:
                        items = "".join([f"<li><b>{w}</b> ({s:.3f})</li>" for w, s in supports_fake])
                        st.markdown(f"<ul style='padding-left:26px;'>{items}</ul>", unsafe_allow_html=True)
                    else:
                        st.write("No strong indicators found.")

                # Only ONE chart: predicted class
                # Decide which label to visualize in LIME HTML so it matches your "supports" list meaning
                # If model predicts Real, show the Fake-side explanation (so scammy words appear under Fake)
                html_label = IDX_FAKE if pred_idx == IDX_REAL else pred_idx

                lime_html = exp.as_html(labels=[html_label], predict_proba=False)

                lime_html_patch = f"""
                {lime_html}
                <script>
                (function() {{
                function boldRealFake() {{
                    const svgs = document.querySelectorAll('svg');
                    svgs.forEach(svg => {{
                    const texts = svg.querySelectorAll('text');
                    texts.forEach(t => {{
                        const s = (t.textContent || "").trim();
                        if (s === "Real" || s === "Fake") {{
                        t.style.fontWeight = "900";
                        }}
                    }});
                    }});
                }}

                // run now + again shortly after (SVG may render slightly later)
                boldRealFake();
                setTimeout(boldRealFake, 50);
                setTimeout(boldRealFake, 200);
                }})();
                </script>
                """

                # ADD THE REMINDER HERE
                st.info(
                    "🔍 **How to read this explanation:**\n\n"
                    "• Words shown under **FAKE** are indicators commonly associated with scam-like postings.\n"
                    "\n• Words shown under **REAL** are indicators commonly associated with legitimate postings.\n\n"
                    "Even if some scam-like words appear, the final decision is based on the **overall balance of all words "
                    "combined with structured features** (e.g. salary patterns, has company profile, telecommuting, etc.).\n\n"
                    "The value may appear as **0.000** due to individual word contributions **extremely small**.\n\n"
                )
                st.session_state["result"]["lime_html_patch"] = lime_html_patch

                lime_html_patch = st.session_state["result"].get("lime_html_patch", None)

                with st.expander("Show interactive LIME explanation (HTML)", expanded=False):
                    if lime_html_patch:
                        st.components.v1.html(lime_html_patch, height=500, scrolling=True)
                    else:
                        st.write("Run prediction with 'Show Explanation' enabled to generate LIME.")




                st.caption(
                    f"LIME HTML shown for label index {html_label} = "
                    f"{'Real' if html_label==IDX_REAL else 'Fake'}"
                )

            except Exception as e:
                st.warning(f"Explanation failed to load: {e}")



        # PDF DOWNLOAD (ONLY ONE)
        lime_for_pdf = res.get("lime_feat_list", lime_feat_list)
        lime_label_idx = res.get("lime_label_idx", IDX_FAKE if pred == 1 else IDX_REAL)

        pdf_file = generate_pdf(res["inputs"], proba_fake, pred, lime_for_pdf, lime_label_idx)

        if pdf_file is not None:
            st.download_button(
                label="📄 Download Report",
                data=pdf_file,
                file_name="job_report.pdf",
                mime="application/pdf",
                key="download_pdf_single"
            )
        else:
            st.info("PDF generation disabled (ReportLab not installed).")

        
with tab2:
    st.header("Bulk CSV Detection")
    st.markdown('<div class="card">Upload a CSV containing job details to process multiple entries at once.</div>', unsafe_allow_html=True)

    # init session state 
    st.session_state.setdefault("bulk_results", None)
    st.session_state.setdefault("bulk_fakes", 0)
    st.session_state.setdefault("bulk_file_bytes", None)  
    st.session_state.setdefault("bulk_just_ran", False)
 

    template_cols = ["title", "location", "department", "salary_range", "company_profile", "description",
                     "requirements", "benefits", "telecommuting", "has_company_logo", "has_questions",
                     "employment_type", "required_experience", "required_education", "industry", "function"]
    template_csv = pd.DataFrame(columns=template_cols).to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download CSV Template",
        data=template_csv,
        file_name="job_detection_template.csv",
        mime="text/csv",
        key="download_csv_template"
    )

    st.info("""
    💡 **Important CSV Requirements:**
    * **Location Format:** `Country, State, City` (e.g., *US, NY, New York*)
    * **Binary Columns:** use **1** for Yes and **0** for No
    * **No Column Name Changes:** follow the template
    """)

    # uploader OUTSIDE form (more stable)
    uploaded = st.file_uploader("Upload CSV", type="csv", key="bulk_csv")
    df_bulk = None

    # save file bytes when user uploads
    # if user clicks X to remove the uploaded file, reset everything
    if uploaded is None and st.session_state["bulk_file_bytes"] is not None:
        st.session_state["bulk_file_bytes"] = None
        st.session_state["bulk_results"] = None
        st.session_state["bulk_fakes"] = 0
        st.session_state["bulk_just_ran"] = False


    # placeholders always created (so layout doesn't jump)
    status_box = st.empty()
    progress_box = st.empty()
    
    run_bulk = st.button("Run Bulk Analysis", key="run_bulk_btn")
    output_box = st.empty()

    # read df from session_state (NOT from uploaded variable)
    # Build df_bulk (robust) 
    df_bulk = None

    # 1) If uploader has a file, read it directly (most reliable)
    if uploaded is not None:
        try:
            new_bytes = uploaded.getvalue()

            # if new file uploaded, reset old results
            if st.session_state.get("bulk_file_bytes") != new_bytes:
                st.session_state["bulk_file_bytes"] = new_bytes
                st.session_state["bulk_results"] = None
                st.session_state["bulk_fakes"] = 0
                st.session_state["bulk_just_ran"] = False

            df_bulk = read_csv_bytes(st.session_state["bulk_file_bytes"])


        except Exception as e:
            status_box.error(f"CSV read error: {e}")
            df_bulk = None


    # 2) Fallback: if uploader is None but we have stored bytes, use bytes
    elif st.session_state.get("bulk_file_bytes") is not None:
        try:
            df_bulk = read_csv_bytes(st.session_state["bulk_file_bytes"])
        except Exception as e:
            status_box.error(f"CSV read error from saved bytes: {e}")
            df_bulk = None

    # normalize column names (only if df exists)
    if df_bulk is not None:
        df_bulk.columns = [c.lower().strip().replace(" ", "_").replace("job_", "") for c in df_bulk.columns]

    # run only when clicked
    if run_bulk:
        if df_bulk is None:
            st.session_state["bulk_just_ran"] = False   
            status_box.error("Please upload a CSV file first.")
        else:
            st.session_state["bulk_just_ran"] = True
            status_box.info("Running bulk analysis... Please wait.")
            pbar = progress_box.progress(0)

            results = []
            fakes = 0
            df_bulk = df_bulk.fillna("Unknown")

            for i in range(len(df_bulk)):
                try:
                    p, _, _ = predict_proba_from_raw_df(df_bulk.iloc[[i]])
                    scr = float(p[0][IDX_FAKE])
                    is_fake = scr >= best_t_xgb
                    if is_fake:
                        fakes += 1

                    results.append({
                        "Job Title": df_bulk.iloc[i].get("title", "N/A"),
                        "Location": df_bulk.iloc[i].get("location", "N/A"),
                        "Suspicion": f"{scr:.2%}",
                        "Verdict": "FAKE" if is_fake else "REAL"
                    })
                except Exception:
                    results.append({
                        "Job Title": f"Error Row {i}",
                        "Location": "Check Data",
                        "Suspicion": "0%",
                        "Verdict": "ERROR"
                    })

                if i % 10 == 0 or i == len(df_bulk) - 1:
                    pbar.progress((i + 1) / len(df_bulk))

            st.session_state["bulk_results"] = pd.DataFrame(results)
            st.session_state["bulk_fakes"] = fakes

            status_box.success("Bulk analysis completed.")
            progress_box.empty()

    # display results always from session_state
    if st.session_state["bulk_results"] is not None:
        res_df = st.session_state["bulk_results"]
        fakes = st.session_state["bulk_fakes"]

        with output_box.container():
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Analyzed", len(res_df))
            c2.metric("Scams Detected", fakes, delta=f"{(fakes/len(res_df)*100):.1f}% Risk", delta_color="inverse")
            c3.metric("Likely Safe", len(res_df) - fakes)

            st.dataframe(res_df, use_container_width=True)
            st.download_button(
                label="📥 Download Results",
                data=res_df.to_csv(index=False).encode("utf-8"),
                file_name="bulk_results.csv",
                mime="text/csv",
                key="download_bulk_results"
            )
        
        # trigger only right after Run Bulk Analysis click
        if st.session_state.get("bulk_just_ran", False):
            if fakes == 0:
                st.balloons()
                st.success("🎉 Excellent! No suspicious postings detected in this batch.")
            elif fakes > (len(res_df) / 2):
                st.warning("⚠️ Caution: More than half of these postings appear suspicious.")

            # turn off so download / cancel upload won't re-trigger
            st.session_state["bulk_just_ran"] = False


        
  
