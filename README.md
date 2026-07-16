# Credit Card Fraud Detection

A machine learning project that compares classical ML models against a deep learning LSTM classifier for detecting fraudulent credit card transactions — with proper handling of extreme class imbalance.

## Live Dashboard

A real-time fraud monitoring dashboard powered by the LSTM model via a Flask API.

**To run locally:**
1. Train the model and save it (`lstm_fraud_model.keras`)
2. `pip install flask flask-cors`
3. `python app.py`
4. Open `dashboard/fraud_dashboard.html` in your browser

**To run in Colab:**
1. Upload `app.py` to Colab
2. `!pip install flask flask-cors pyngrok -q`
3. `!python app.py` → copy the ngrok URL
4. Paste URL into `API_URL` in the dashboard HTML


---

## Models Compared

| Model | Accuracy | Note |
|---|---|---|
| Logistic Regression | 99.87% | Misleading — see below |
| Decision Tree | 99.93% | Misleading — see below |
| Random Forest | 99.95% | Misleading — see below |
| SVM | 99.73% | Misleading — see below |
| **LSTM (Deep Learning)** | — | Uses `class_weight` to handle imbalance |

> **Why accuracy is misleading here:** The dataset has ~0.17% fraud rate. A model that predicts *every transaction as normal* would get 99.83% accuracy — without detecting a single fraud. Always evaluate fraud detection using **Precision, Recall, and F1-score** on the fraud class.

---

## Dataset

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle.

- 284,807 transactions, 492 fraud cases (~0.17%)
- Features V1–V28 are PCA-transformed (anonymized)
- Raw features: `Time`, `Amount`, `Class`

**Download the dataset from Kaggle and place it at:**
```
data/creditcard.csv
```


---


## Key Concepts

**Class Imbalance**
Fraud datasets are heavily skewed. This project addresses it in the LSTM using `class_weight`, which penalizes missed fraud predictions proportionally to the imbalance ratio (~580×).

**Why LSTM on tabular data?**
LSTM treats each transaction as a timestep sequence, capturing temporal patterns in spending behavior. While the standard `creditcard.csv` doesn't include card-level grouping, this architecture is designed to scale to real-world data where sequences per card are available.

**Evaluation Metrics that matter**
- Fraud **Recall** → how many actual frauds were caught
- Fraud **Precision** → of flagged transactions, how many were real fraud
- **ROC-AUC** → overall discrimination ability

---

## Tech Stack

- Python, Pandas, NumPy
- scikit-learn 
- TensorFlow / Keras 
  

---

## Future Work

- Group transactions by card ID for true sequence modeling
- LSTM Autoencoder (unsupervised anomaly detection — no labels needed)
- Threshold tuning via Precision-Recall curve for business cost optimization
