# Fake Job Posting Detection Using Explainable NLP Models

## Overview

Fake Job Posting Detection Using Explainable NLP Models is a machine learning and natural language processing (NLP) project designed to identify fraudulent job advertisements automatically. The system combines textual information and structured job features to classify job postings as **Real** or **Fake** while providing transparent explanations using Explainable AI (LIME).

The application is deployed as an interactive Streamlit web application and supports both single job analysis and bulk CSV prediction.

---

## Features

* Single Job Posting Detection
* Bulk CSV Detection
* TF-IDF + XGBoost classification model
* Integration of structured job features
* Optimized decision threshold for fraud detection
* Explainable AI using LIME
* PDF report generation
* Downloadable prediction results
* User-friendly Streamlit interface

---

## Dataset

This project uses the **Real or Fake Job Posting Prediction** dataset from Kaggle.

Dataset characteristics:

* Total records: 17,014 job postings
* Fake postings: 866 (approximately 4.84%)
* Binary target variable:

  * 0 = Real
  * 1 = Fake

The dataset contains both:

### Textual Features

* Job title
* Company profile
* Description
* Requirements
* Benefits

### Structured Features

* Location
* Employment type
* Required experience
* Required education
* Company logo
* Telecommuting option
* Salary information

---

## Methodology

### Data Preparation

* Remove duplicate records
* Handle missing values
* Text preprocessing
* Feature engineering
* Salary conversion and normalization
* Location extraction
* Creation of binary indicators

### NLP Techniques Evaluated

#### TF-IDF + Classical Machine Learning

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* XGBoost

#### Word2Vec Embeddings

* Word2Vec + Logistic Regression

#### Transformer-Based Model

* DistilBERT

---

## Handling Class Imbalance

Several techniques were used:

* Stratified train-validation-test split
* Class weighting
* Threshold tuning
* Precision-Recall AUC optimization

---

## Model Performance

| Model                          | F1 Score | PR-AUC |
| ------------------------------ | -------: | -----: |
| TF-IDF + XGBoost               |    0.845 |  0.907 |
| DistilBERT                     |    0.825 |  0.878 |
| TF-IDF + Logistic Regression   |    0.777 |  0.858 |
| TF-IDF + Random Forest         |    0.746 |  0.845 |
| TF-IDF + SVM                   |    0.588 |  0.646 |
| Word2Vec + Logistic Regression |    0.572 |  0.556 |

---

## Final Deployed Model

### TF-IDF + XGBoost

Final test performance:

* F1-score: **0.869**
* PR-AUC: **0.931**

The model was selected because of its:

* High fraud detection capability
* Stability under class imbalance
* Fast inference speed
* Compatibility with Explainable AI techniques
* Suitability for deployment

---

## Explainable AI Using LIME

LIME (Local Interpretable Model-Agnostic Explanations) is used to explain individual predictions.

The explanation highlights:

* Words contributing toward Fake predictions
* Words contributing toward Real predictions

This improves transparency and user trust.

---

## Streamlit Web Application

The web application contains two modules.

### Single Job Check

Users can:

* Enter job title and description
* Fill optional structured features
* Predict whether a posting is Real or Fake
* View fraud probability score
* Generate LIME explanations
* Download PDF reports

### Bulk CSV Detection

Users can:

* Upload multiple job postings
* Run batch analysis
* View fraud statistics
* Download prediction results in CSV format

---

## Technologies Used

### Programming Language

* Python

### Machine Learning Libraries

* Scikit-learn
* XGBoost
* Gensim
* Transformers

### Explainable AI

* LIME

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Web Framework

* Streamlit

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/fake-job-detector.git
```

Move into the project directory:

```bash
cd fake-job-detector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## Future Improvements

* Incorporate external company verification
* Apply SHAP explanations for structured features
* Experiment with larger transformer models
* Improve detection of rare fraud patterns
* Integrate with online recruitment platforms

---

## Author

**Heng Xin Phei**

Faculty of Computer Science and Information Technology
University of Malaya

Email:

[23005228@siswa.um.edu.my](mailto:23005228@siswa.um.edu.my)

---

## Supervisor

**Associate Professor Dr. Tutut Herawan**

Faculty of Computer Science and Information Technology

University of Malaya

---
