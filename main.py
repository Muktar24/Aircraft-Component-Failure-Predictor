import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from app import X_train, X_train_scaled
from backend import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#--Load data---
df=pd.read_csv("data/component_data.csv")
X=df.drop("Failure Probability")
y=df["Failure Probability"]

X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)

theta=gradient_descent(X_train_scaled,Y_train,lr=0.1)

def app():
    st.set_page_config(page_title="Aircraft Failure Predictor",page_icon="✈️",layout="wide")
    st.title("✈️ Aircraft Component Failure Prediction Dashboard")
    st.markdown("Predict Components failure using Logistic algorithm")

    st.sidebar.header("Input Componenent Parameters")
    features ={
        "Temperature (°C)": st.sidebar.slider("Temperature (°C)", 200, 900, 700),
        "Pressure (psi)": st.sidebar.slider("Pressure (psi)", 100, 400, 300),
        "Vibration (g)": st.sidebar.slider("Vibration (g)", 0.0, 1.0, 0.3),
        "Oil Quality (%)": st.sidebar.slider("Oil Quality (%)", 10, 100, 80),
        "Flight Hours": st.sidebar.slider("Flight Hours", 0, 5000, 1500),
        "Ambient Temp (°C)": st.sidebar.slider("Ambient Temp (°C)", -20, 50, 25),
        "Altitude (ft)": st.sidebar.slider("Altitude (ft)", 0, 15000, 10000),
        "Last Service Cycles": st.sidebar.slider("Last Service Cycles", 0, 500, 150)

    }

    X_input =pd.DataFrame([features],columns=X.columns)
    X_inserted_scaled=scaler.transform(X_input)
    prob=predict_prob(X_inserted_scaled,theta)[0]
    st.subheader("Prediction Results")
    st.metric("Failure Probability",f"{prob *100:.2f}%")
    if prob >= 0.6:
        st.error("⚠️ High Risk: Component likely to FAIL soon!")
    else:
        st.success("✅ Low Risk: Component operating normally.")