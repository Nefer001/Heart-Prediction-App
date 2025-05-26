import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

st.set_page_config(page_title='Heart Disease Prediction App', layout='wide')
# Load data
df = pd.read_csv('heart.csv')

# Sidebar Navigation
st.sidebar.subheader('🧭 Navigation Pages')
page = st.sidebar.radio("Select Page", ['Introduction', 'Data Overview', 'Modeling Results', 'Prediction'])

# Page: Introduction
if page == 'Introduction':
    st.title("Heart Disease Prediction App")
    st.markdown("""
        This app uses Machine Learning to predict whether a person has heart disease based on medical data.
        \nDataset Source: [Heart Disease UCI](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
    """)

# Page: Data Overview
elif page == 'Data Overview':
    st.header("Dataset Overview")
    st.dataframe(df.head())

    st.subheader("Basic Info")
    st.write(df.describe())

    st.subheader("Class Distribution")
    st.bar_chart(df['HeartDisease'].value_counts())

# Page: Modeling Results
elif page == 'Modeling Results':
    st.header("Model Evaluation")

    # Encode categorical features
    df_encoded = df.copy()
    cat_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    for col in cat_cols:
        df_encoded[col] = pd.factorize(df_encoded[col])[0]

    X = df_encoded.drop('HeartDisease', axis=1)
    y = df_encoded['HeartDisease']

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = RandomForestClassifier()
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    st.subheader("Accuracy Score")
    st.write(f"{model.score(X_scaled, y) * 100:.2f}%")

    st.subheader("Classification Report")
    report = classification_report(y, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Feature Importance")
    importance = model.feature_importances_
    feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
    feat_df = feat_df.sort_values(by='Importance', ascending=False)

    fig, ax = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=feat_df, ax=ax)
    st.pyplot(fig)

# Page: Prediction
elif page == 'Prediction':
    st.header("Predict Heart Disease")

    df_encoder = LabelEncoder()
    df['Sex'] = df_encoder.fit_transform(df['Sex'])
    df['ChestPainType'] = df_encoder.fit_transform(df['ChestPainType'])
    df['RestingECG'] = df_encoder.fit_transform(df['RestingECG'])
    df['ExerciseAngina'] = df_encoder.fit_transform(df['ExerciseAngina'])
    df['ST_Slope'] = df_encoder.fit_transform(df['ST_Slope'])
    # User Input
    age = st.slider("Age", 20, 80, 40)
    sex = st.selectbox("Sex", ['M', 'F'])
    cp = st.selectbox("Chest Pain Type", ['TA', 'ATA', 'NAP', 'ASY'])
    restBP = st.number_input('RestingBP', 90, 170, 100)
    resting_ecg = st.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.radio("Fasting Blood Sugar > 120", [0, 1])
    hr = st.slider("Max Heart Rate", 60, 202, 150)
    angina = st.radio("Exercise-induced Angina", ['Y', 'N'])
    st_slope = st.selectbox("ST Slope", ['Up', 'Flat', 'Down'])
    oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)

    # Map inputs to model format
    sex = 1 if sex == 'M' else 0
    cp_map = {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3}
    resting_map = {'Normal': 1, 'ST': 2, 'LVH': 0}
    angina = 1 if angina == 'Y' else 0
    slope_map = {'Up': 2, 'Flat': 1, 'Down': 0}

    sample = pd.DataFrame({
        'Age':[age], 'Sex':[sex], 'ChestPainType':[cp_map[cp]], 'RestingBP':[restBP], 'Cholesterol':[chol], 'FastingBS':[fbs],'RestingECG':[resting_map[resting_ecg]], 'MaxHR':[hr], 'ExerciseAngina':[angina],
        'Oldpeak':[oldpeak], 'ST_Slope':[slope_map[st_slope]]
    })

    #Overall Model Training
    X = df.drop('HeartDisease', axis=1)
    Y = df['HeartDisease']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier()
    model.fit(X_scaled, Y)

    # Predict
    input_scaled = scaler.transform(sample)
    prediction = model.predict(input_scaled)
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error("⚠️High Risk: Person may have heart disease.")
    else:
        st.success("✅Low Risk: Person is unlikely to have heart disease.")
