# ✈️ Aircraft Component Failure Prediction Dashboard

A data-driven **aircraft maintenance intelligence system** that predicts the **failure probability of aircraft components** using a **logistic regression model** trained on simulated aviation sensor data.  
Built with **Streamlit**, this dashboard provides interactive visualizations, real-time predictions, and feature importance insights to support predictive maintenance decisions.

---

## 🚀 Features

✅ **Interactive Prediction Panel**
- Adjust key operating parameters (temperature, pressure, vibration, oil quality, etc.)
- Instantly get the predicted **failure probability** of a component

✅ **Data Visualization**
- See how different parameters (like temperature or oil quality) affect component reliability
- Compare normal vs failed component behavior using KDE plots

✅ **Model Insights**
- Displays feature importance to show which factors most influence failure
- Visualize learned coefficients of the logistic regression model

✅ **Report Generation**
- Download personalized prediction reports in CSV format

✅ **Stylish & Responsive UI**
- Built using **Streamlit** with custom layout, color palette, and intuitive design
- Footer and sidebar for smooth navigation

---

## 📊 Parameters Used

| Parameter | Description | Example Range |
|------------|--------------|----------------|
| Temperature (°C) | Operating component temperature | 200 – 900 |
| Pressure (psi) | Internal system pressure | 100 – 400 |
| Vibration (g) | Component vibration intensity | 0.0 – 1.0 |
| Oil Quality (%) | Remaining lubrication quality | 0 – 100 |
| Flight Hours | Total operating hours since last overhaul | 0 – 5000 |
| Ambient Temp (°C) | Surrounding air temperature | -20 – 50 |
| Altitude (ft) | Aircraft operating altitude | 0 – 15000 |
| Last Service Cycles | Cycles since last maintenance service | 0 – 500 |

---

## 🧠 Model Overview

The model uses **logistic regression** to classify whether a component is likely to fail or operate normally.

### Model Equation



**[P(Failure) = 1 / (1 + exp(-(θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ)))]()**


- Parameters (`θ`) are learned using **gradient descent**
- The dataset used for training is a **synthetic but realistic dataset** with 20,000+ samples
- Output: Continuous **Failure Probability (0–1)**

---

## ⚙️ Tech Stack

- **Python 3.10+**
- **Streamlit** — Frontend dashboard
- **Pandas / NumPy** — Data manipulation
- **Matplotlib / Seaborn** — Data visualization
- **Scikit-learn** — Preprocessing & scaling
- **Custom Logistic Regression (from scratch)** — For model training
- **Joblib** — Model persistence (optional)

---

## 🧩 Project Structure

* 📦 Aircraft_Failure_Predictor
* 
* ├── app.py # Main Streamlit application
- ├── backend.py # Logistic regression functions (sigmoid, training, etc.)
* ├── pre_processing.py # Data loading, normalization, preprocessing
- ├── data/
+ │ └── component_data.csv # Synthetic training dataset
- └── README.md # Project documentation

## 👨‍💻 Developer

[- **Sanusi Muktar**]()
- Aircraft Engineer |ML & AI Engineer | Python Developer
- 📍 Moscow Aviation Institute (MAI)
- [- ✉️ sanusimuktar29@gmail.com]()

