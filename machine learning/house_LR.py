import numpy as np
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --- Train model on generated data ---
np.random.seed(42)
n = 500
area      = np.random.randint(100, 4000, n)
bedrooms  = np.random.randint(1, 6, n)
age       = np.random.randint(0, 15, n)
price     = area * 0.038 + bedrooms * 4.2 - age * 0.9 

X = np.column_stack([area, bedrooms, age])
y = price

X_mean = np.mean(X, axis=0)
X_std  = np.std(X, axis=0)
X_norm = (X - X_mean) / X_std

model = LinearRegression()
model.fit(X_norm, y)

# --- Take input from user ---
print("=" * 45)
print("   🏠 HOUSE PRICE PREDICTOR")
print("=" * 45)


st.title("🏠 House Price Predictor")


area_in  = st.number_input("Area (sq ft)",        min_value=1,  max_value=20000, value=1500)
beds_in  = st.number_input("Bedrooms",             min_value=1,  max_value=10,    value=3)
age_in   = st.number_input("Age of house (years)", min_value=0,  max_value=100,   value=5)

user_X     = np.array([[area_in, beds_in,  age_in]])
user_X_norm= (user_X - X_mean) / X_std
predicted  = model.predict(user_X_norm)[0]
if st.button("🔍 Predict Price"):
    # prediction sirf button dabane pe ho
     st.metric("Predicted Price", f"₹{predicted:.2f}L")
 
