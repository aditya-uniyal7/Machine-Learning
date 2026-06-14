import numpy as np
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("House_price.csv")

X = df[["Area", "Number of Bedrooms", "House Age"]].values
y = df["Price"].values

X_mean = np.mean(X, axis=0)
X_std  = np.std(X, axis=0)
X_norm = (X - X_mean) / X_std

model = LinearRegression()
model.fit(X_norm, y)

st.title("🏠 House Price Predictor")

area_in = st.number_input("Area (sq ft)",        min_value=1, value=1500)
beds_in = st.number_input("Bedrooms",             min_value=1, value=3)
age_in  = st.number_input("Age of house (years)", min_value=0, value=5)

if st.button("🔍 Predict Price"):
    user_X      = np.array([[area_in, beds_in, age_in]])
    user_X_norm = (user_X - X_mean) / X_std
    predicted   = model.predict(user_X_norm)[0]
    st.metric("💰 Predicted Price", f"₹{predicted:.2f}L")

from sklearn.metrics import r2_score

y_pred = model.predict(X_norm)
r2 = r2_score(y, y_pred)

st.metric("R² Score", f"{r2:.2f}")
