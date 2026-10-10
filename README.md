# CyberGuard AI - Project Overview

## 1. What Your Project Is
My project is **CyberGuard AI**, an advanced machine learning pipeline designed to automatically detect and classify cyberbullying in online text (specifically tweets). It categorizes social media text into specific forms of bullying such as age, gender, ethnicity, and religion. It serves as a rapid, interpretable content moderation system.

## 2. Technologies Used
I have built this using a robust Python ecosystem for data science and web development:
*   **Web Framework:** `streamlit` (for the interactive frontend UI).
*   **Data Processing:** `pandas`, `numpy`.
*   **Machine Learning Core:** `scikit-learn` (for base models, TF-IDF, pipelines), `xgboost` (meta-learner).
*   **Handling Imbalanced Data:** `imbalanced-learn` (SMOTE).
*   **Natural Language Processing (NLP):** `nltk` (tokenization, lemmatization), `vaderSentiment` (sentiment profiling), `emoji` (demojizing).
*   **Data Visualization:** `plotly` (interactive charts in the UI).
*   **Model Persistence:** `joblib` (for saving and loading the trained pipeline).

## 3. My Architecture
I have implemented a **Stacking Ensemble Architecture**:
*   **Feature Engineering (The Pipeline):** Your model takes raw text and applies multiple custom transformations. It uses a `ColumnTransformer` combining Word-level TF-IDF, Character-level TF-IDF (to catch misspelled insults), Masked text TF-IDF, and VADER numeric sentiment scores.
*   **Base Estimators:** You are using three diverse traditional ML models to process these features:
    1.  Calibrated Linear SVC (Support Vector Classifier)
    2.  Logistic Regression
    3.  Complement Naive Bayes
*   **Meta-Learner:** The probabilistic outputs of the three base models are fed into an **XGBoost Classifier**, which acts as the meta-learner to make the final, highly accurate prediction.

## 4. Main Ideology / Core Innovation
The core ideology is to provide a **computationally lightweight and highly interpretable alternative to slow, expensive "black-box" Deep Learning models (like BERT)**. 

I achieved this by substituting heavy neural networks with intelligent, domain-specific feature engineering. Key innovations include:
*   **Class-Aware Demographic Proximity:** Calculating the syntactic distance between demographic keywords (e.g., "muslim", "woman") and toxic insults to understand *who* the insult is directed at.
*   **Interaction Masking:** Distinguishing between the "Target" (you, them) and the "Self" (I, me) to prevent false positives in self-deprecating humor.
*   **Text Normalization:** Decoding common obfuscation techniques (like "@sshole").

## 5. Main Files Identified
*   **Main UI File:** `app2.py` is your most comprehensive Streamlit frontend file. It contains the animated UI, data dashboard, and inference logic. 
*   **Main ML Pipeline File:** `ml2.py` is your core machine learning script that trains the Stacking Classifier on `cyberbullying_tweets.csv` and generates the `ml2_pipeline.joblib` file used by the frontend.
*   *(Note: Older or alternate versions like `app.py`, `ml.py`, etc. have been safely moved to the `trial_versions` folder).*

## 6. Results
The configured Stacking Ensemble model demonstrates exceptional performance, rivaling results typically achieved by much heavier Transformer-based architectures. 
*   It effectively minimizes false positives by recognizing the difference between a self-deprecating statement and a targeted offensive attack.
*   It achieves highly balanced F1-scores across all difficult categories (Age, Ethnicity, Gender, Religion).
*   It maintains sub-second inference times and low memory overhead, making it viable for live social media APIs edge deployment.

## 7. How to Run Your Project
To run the application, open your terminal (PowerShell) in the project directory and follow these steps:

1.  **Activate your virtual environment**:
    ```powershell
    .\.venv\Scripts\activate
    ```
2.  **Install requirements** (if you haven't already):
    ```powershell
    pip install -r requirements.txt
    ```
3.  **Run the Main Application:**
    ```powershell
    streamlit run app2.py
    ```
