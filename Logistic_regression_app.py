import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load and preprocess dataset
df = pd.read_csv("diabetes.csv")
col_missing_values = ['Pregnancies', 'Glucose', 'BloodPressure', 
                      'SkinThickness', 'Insulin', 'BMI']
df[col_missing_values] = df[col_missing_values].replace([0, None], np.nan)
df[col_missing_values] = df[col_missing_values].apply(pd.to_numeric)
for col in col_missing_values:
    df[col].fillna(df[col].mean(), inplace=True)

X = df.drop(columns=['Outcome'])
Y = df['Outcome']

SS = StandardScaler()
SS_X = SS.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(SS_X, Y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, Y_train)

# Streamlit UI
st.title("Diabetes Prediction App")

st.write("Enter patient details to predict diabetes outcome:")

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=30)

# Collect inputs
input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
                          columns=X.columns)

# Scale input
input_scaled = SS.transform(input_data)

# Prediction
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

st.subheader("Prediction Result")
if prediction == 1:
    st.write("⚠️ The patient is likely to have diabetes.")
else:
    st.write("✅ The patient is unlikely to have diabetes.")

st.write(f"Probability of diabetes: {probability:.2f}")