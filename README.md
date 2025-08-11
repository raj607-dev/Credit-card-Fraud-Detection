# 💳 Credit Card Fraud Detection using Python & Machine Learning

## 📌 Project Overview
Credit card fraud detection is a critical task for banks and financial institutions to protect customers from unauthorized transactions.  
In this project, we develop a **binary classifier** to predict whether a given transaction is fraudulent or genuine.

This project explores **data preprocessing, exploratory data analysis (EDA), class imbalance handling**, and applies **Machine Learning algorithms** such as **Decision Trees** and **Random Forests**.

---

## 📂 Dataset
- **Source:** [Anonymized Credit Card Transactions for Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Shape:** `284,807` transactions  
- **Fraud Cases:** Only `492` (~0.17%), causing **severe class imbalance**  
- **Features:** 31 total  
  - 28 anonymized features from **PCA transformation**
  - `Time` (removed during preprocessing)  
  - `Amount` (normalized)  
  - `Class` (0 = Genuine, 1 = Fraud)  

---

## 🛠 Tools & Libraries
- **Python** 3.x
- **NumPy** 1.19.2
- **Pandas**
- **Scikit-learn** 0.24.1
- **Matplotlib** 3.3.4
- **Imbalanced-learn (imblearn)** 0.8.0
- **Collections & Itertools**

---

## 📊 Exploratory Data Analysis (EDA)

### Class Distribution
Genuine transactions dominate the dataset.  
Fraud accounts for **less than 1%**.

**Bar chart of class distribution:**
![Class Imbalance](images/class_imbalance.png)

---

## 🔄 Data Preprocessing
1. **Checked for null values** (none found ✅)
2. **Normalized the `Amount` feature** using `StandardScaler`
3. **Dropped irrelevant columns** (`Time`, original `Amount`)
4. **Split dataset** into train (70%) and test (30%)

---

## 🤖 Machine Learning Models

### Models Used:
- **Decision Tree Classifier**
- **Random Forest Classifier**

Both models were trained and evaluated.  
The **Random Forest** showed better performance on imbalanced data.

---

## 📈 Model Performance (Before Handling Imbalance)

| Model          | Accuracy | Precision | Recall | F1-score |
|----------------|----------|-----------|--------|----------|
| Decision Tree  | ~99%     | ~71%      | ~67%   | ~69%     |
| Random Forest  | ~99.9%   | ~96%      | ~88%   | ~92%     |

---

## ⚖ Handling Class Imbalance
To address the imbalance, we used **SMOTE (Synthetic Minority Oversampling Technique)** from `imblearn`.

After applying SMOTE:
- Balanced dataset: equal number of fraud and genuine transactions
- Retrained **Random Forest Classifier**
- Achieved **higher recall & F1-score**

**Confusion Matrix After SMOTE:**
![Confusion Matrix After SMOTE](images/confusion_matrix_smote.png)

---

## 🏆 Final Results
The **Random Forest Classifier** after SMOTE achieved:
- **High Accuracy**
- **Improved Recall** for fraud detection
- Balanced performance across metrics

---

## 📂 Project Structure

