import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title("Diabetes Prediction App")

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(x_train, y_train)

bmi = st.number_input("Enter BMI value")

if st.button("Predict"):
    sample = x_test[0].copy()
    sample[2] = bmi
    
    prediction = model.predict([sample])
    st.write("Prediction:", prediction[0])
