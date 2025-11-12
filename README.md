# Intent Prediction using Gradient Boosting

## Overview
This project predicts **online shopper intent** (e.g., purchase vs. no purchase) using behavioral and categorical data from web sessions. It involves preprocessing, feature encoding, model training, and evaluation using ensemble learning methods.

---

## Dataset
The dataset used is **`online_shoppers_intention.csv`**, containing anonymized web session data.  
Each row represents a unique session with user behavior and contextual features.

### Key Columns
- `UserCategory`: Type of visitor (e.g., Returning, New, Other)
- `VisitMonth`: Month of visit (Jan–Dec)
- Numerical and categorical features indicating browsing behavior, product views, and purchase actions.

---

## Project Structure
| Step | Description |
|------|--------------|
| **1. Data Cleaning** | Handled missing values, inspected types, and verified feature consistency. |
| **2. Data Normalization** | Standardized numeric features and encoded categorical ones. |
| **3. Model Selection** | Compared models; selected Gradient Boosting for its strong performance. |
| **4. Oversampling** | Addressed class imbalance using oversampling of minority samples. |
| **5. Model Evaluation** | Measured model accuracy and tested weighting strategies on the minority class. |

---

## Model
**Gradient Boosting Classifier** from `scikit-learn` is used for classification.

### Techniques Applied
- One-hot encoding for categorical columns
- Ordinal encoding for ordered variables like months
- Oversampling using techniques such as `SMOTE` (if applied)
- Evaluation using metrics like precision, recall, F1-score

---

## Requirements
```bash
pandas
numpy
scikit-learn
matplotlib
seaborn
```