# 🫀 Heart Disease Risk Prediction App (Streamlit + Machine Learning)

This project is a **deployed Streamlit web app** that predicts whether a patient is at **low risk or high risk** of heart disease based on medical parameters. It uses supervised machine learning — **Logistic Regression** and **Random Forest** — to classify risk levels with high accuracy.

---

##  Live Demo

🔗 [Streamlit App Link](https://your-streamlit-app-url)

---

## 📊 Dataset Information

- Source: [Kaggle – Heart Disease UCI Dataset](https://www.kaggle.com/datasets/ronitf/heart-disease-uci)
- Total Records: 303
- Key Features:
  - Age, Sex, Chest Pain Type (`cp`)
  - Blood Pressure (`trestbps`), Cholesterol (`chol`)
  - Fasting Blood Sugar (`fbs`), Rest ECG (`restecg`)
  - Max Heart Rate (`thalach`), Exercise Angina (`exang`)
  - ST Depression (`oldpeak`), ST Slope (`slope`)
  - Major Vessels (`ca`), Thalassemia (`thal`)
- Target: `1` = High Risk, `0` = Low Risk

---

##  Project Workflow

### ✅ 1. Data Exploration & Preprocessing
- Null value checks, datatype fixes
- Categorical encoding
- Feature scaling with `StandardScaler`

### ✅ 2. Model Building
- **Logistic Regression**: for baseline
- **Random Forest**: for better accuracy & feature insight

### ✅ 3. Evaluation Metrics
- Accuracy, Confusion Matrix, Classification Report
- Random Forest chosen for Streamlit deployment due to higher performance

### ✅ 4. Streamlit App
- Built an interactive web app where users input patient data
- Outputs instant prediction: **Low Risk** or **High Risk**
- Displays key features and highlights prediction interpretation

---

## 📈 Results Summary

| Model              | Accuracy |
|-------------------|----------|
| Logistic Regression | ~85%     |
| Random Forest       | ~90%     ✅

---

##  Tech Stack

- Python, Pandas, NumPy, scikit-learn
- Streamlit for interactive dashboard
- Matplotlib & Seaborn for visuals

---

## 🧠 Future Plans

- Add more model options (e.g. XGBoost, LightGBM)
- Add SHAP explanation for model interpretability
- Improve UI with animations and user-friendly inputs

---

##  Author

**[EDWARD DEODATUS]**  
🔗 GitHub: [github.com/Nefer001](https://github.com/Nefer001)  
📫 Upwork: [Your Upwork profile link]

---

##  License

This project is licensed under the MIT License.
