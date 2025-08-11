# 💳 Credit Card Fraud Detection with Python & Machine Learning

## 📌 Overview
This project implements **Credit Card Fraud Detection** using **Python** and various **Machine Learning algorithms**.  
The goal is to accurately classify transactions as **fraudulent** or **genuine**, even in the presence of severe **class imbalance**.

We explore multiple algorithms (Decision Tree, Random Forest) and address imbalance using **SMOTE oversampling** to improve model performance.

---

## 📂 Dataset
- **Source:** [Anonymized Credit Card Transactions for Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Total Transactions:** 284,807  
- **Fraudulent Transactions:** 492 (~0.17%)  
- **Features:** 31 (28 PCA-transformed, plus `Time` and `Amount`)

⚠ Due to confidentiality, most features are anonymized via **PCA transformation**.

---

## 🛠 Tools & Libraries Used
- Python 3.x
- **Numpy** – 1.19.2  
- **Pandas** – 1.x  
- **Scikit-learn** – 0.24.1  
- **Matplotlib** – 3.3.4  
- **Imbalanced-learn (imblearn)** – 0.8.0  
- Collections, Itertools  

---

## 🔍 Project Workflow

### 1️⃣ Exploratory Data Analysis (EDA)
- Checked for missing values (none found ✅)
- Analyzed **class distribution** → Found strong imbalance (>99% genuine transactions)
- Scaled `Amount` feature using `StandardScaler`
- Dropped `Time` column (irrelevant)

📊 **Class Distribution Plot:**  
![Class Distribution](images/class_distribution.png)

---

### 2️⃣ Data Preprocessing
- Normalized `Amount` → `NormalizedAmount`
- Split dataset into **train (70%)** and **test (30%)** sets
- Addressed class imbalance using **SMOTE oversampling**

📊 **After SMOTE:**  
Balanced classes with equal number of fraudulent and genuine samples.

---

### 3️⃣ Model Building
Implemented:
- **Decision Tree Classifier**
- **Random Forest Classifier**

📌 Random Forest slightly outperformed Decision Tree before and after applying SMOTE.

---

### 4️⃣ Model Evaluation
Metrics used:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

#### 📊 Confusion Matrix - Decision Tree (Before SMOTE)
![Confusion Matrix DT](images/confusion_matrix_dt.png)

#### 📊 Confusion Matrix - Random Forest (Before SMOTE)
![Confusion Matrix RF](images/confusion_matrix_rf.png)

#### 📊 Confusion Matrix - Random Forest (After SMOTE)
![Confusion Matrix RF SMOTE](images/confusion_matrix_rf_smote.png)

---

## 📈 Results
| Model               | Accuracy | Precision | Recall | F1-score |
|---------------------|----------|-----------|--------|----------|
| Decision Tree       | XX%      | XX%       | XX%    | XX%      |
| Random Forest       | XX%      | XX%       | XX%    | XX%      |
| Random Forest + SMOTE | XX%    | XX%       | XX%    | XX%      |

✅ **Random Forest with SMOTE achieved the best balance between Recall and Precision**, making it more suitable for fraud detection where catching fraudulent cases is crucial.

---

## 🚀 How to Run the Project
```bash
# Clone the repository
git clone https://github.com/yourusername/credit-card-fraud-detection.git

# Navigate to the project directory
cd credit-card-fraud-detection

# Install dependencies
pip install -r requirements.txt

# Run the script
python credit_card_fraud_detection.py


