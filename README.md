# 💗 Breast Cancer Classification Using Machine Learning

This project uses various supervised machine learning models to predict whether a tumor is **Malignant (Cancerous)** or **Benign (Non-Cancerous)** based on features extracted from breast cancer cell images.

## 🖼️ App Screenshot / Jupyter Notebook Screenshot
![image](https://github.com/user-attachments/assets/bb39d376-e016-4c66-8a9a-4c592005dde3)
![image](https://github.com/user-attachments/assets/2e9e4749-c52e-4530-a9c9-0ede3790e5fa)
![image](https://github.com/user-attachments/assets/275b66b7-4a1c-44c7-8bae-37f4cd56ed70)
![image](https://github.com/user-attachments/assets/7a29a5e2-b907-4c40-81e6-eec7846cd717)
![image](https://github.com/user-attachments/assets/b58e1aa5-b7e1-40a0-97ce-255fd5b6ba58)
![image](https://github.com/user-attachments/assets/748f05e5-be7e-49e5-9ab5-545c5a08c731)
![image](https://github.com/user-attachments/assets/67d604d4-2af2-487d-a350-8f32abaef5d0)


## 🔬 Problem Statement

Breast cancer is one of the leading causes of cancer-related deaths among women. Early diagnosis significantly increases the chances of successful treatment. This project uses machine learning techniques to classify tumors using the **Breast Cancer Wisconsin (Diagnostic) Dataset**.

---

## 📁 Project Structure

```bash
📦 Breast Cancer Classification Using ML
├── breast_cancer_app/         # Streamlit web app
│   ├── app.py                 # Main app script
│   ├── xgboost_breast_cancer_model.pkl
│   └── scaler.pkl
├── dataset/
│   └── breast_cancer.csv      # Raw dataset
├── notebook/
│   └── breast_cancer_model.ipynb  # Jupyter notebook (EDA + training)
└── README.md
🧠 Machine Learning Models Used
✅ Logistic Regression

✅ K-Nearest Neighbors (KNN)

✅ Support Vector Machine (SVM)

✅ Decision Tree

✅ Random Forest

✅ Gradient Boosting Classifier

✅ XGBoost Classifier ✅ (Best performer)

📊 Model Performance Summary
Model	Train Accuracy	Test Accuracy	Recall (Malignant)
XGBoost	100%	97.37%	97.87%
Gradient Boosting	100%	97.37%	97.87%
Random Forest	100%	96.49%	93.62%
KNN	92.31%	96.49%	97.87%
Logistic Regression	95.38%	95.61%	93.62%
Decision Tree	100%	92.98%	95.74%
SVM	90.99%	92.98%	85.11%

📌 XGBoost was selected as the final model for deployment.

🌐 Web Application
An interactive Streamlit app allows users to input tumor features and get real-time predictions.

🚀 Run the App Locally
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/breast-cancer-classification.git
cd breast-cancer-classification/breast_cancer_app
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py


💾 Files Included
breast_cancer_model.ipynb – Full data preprocessing, EDA, feature selection, and model training.

scaler.pkl – Pre-trained scaler for transforming inputs.

xgboost_breast_cancer_model.pkl – Final trained model.

app.py – Streamlit interface for real-time prediction.


📌 Key Learnings
End-to-end ML project development

Feature correlation analysis and reduction

Model comparison and evaluation

Model deployment with Streamlit

Saving and loading models using joblib
