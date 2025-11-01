import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from backend import *
from pre_processing import *
from sklearn.preprocessing import StandardScaler

# --- Load Data ---
df = pd.read_csv("./Data/component_data.csv")
X = df.drop(["Failure Probability"], axis=1)
y = df["Failure Probability"]
X_train, y_train = load_and_preprocess()

# --- Scale and Train ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
theta = gradient_descent(X_train_scaled, y_train)

def app():
    
    if "first_visit" not in st.session_state:
        st.session_state.first_visit = True
    
    else:
        st.session_state.first_visit = False  
# Inject CSS for blinking effect
    if st.session_state.first_visit:
        st.markdown("""
        <style>
        /* Target the specific sidebar toggle button */
        [data-testid="baseButton-header"] {
            animation: blink 1s infinite;
            border: 2px solid #ff4b4b !important;
            border-radius: 5px;
        }
    
        @keyframes blink {
            0% { box-shadow: 0 0 10px 2px #ff4b4b; }
            50% { box-shadow: 0 0 0 0 #ff4b4b; }
            100% { box-shadow: 0 0 10px 2px #ff4b4b; }
        }
        </style>
        """, unsafe_allow_html=True)
    # --- App Setup ---
    
    st.set_page_config(page_title="Aircraft Failure Predictor", page_icon="✈️", layout="wide")
    st.markdown(
    """
    <style>
    /* Blinking animation keyframes */
    @keyframes blink {
        50% {
            opacity: 0.3;
        }
    }

    /* Select the top-left menu button in Streamlit */
    button[data-testid="baseButton-headerNoPadding"] {
        animation: blink 1s infinite;
        border: 2px solid #ff4b4b !important;
        border-radius: 8px;
        box-shadow: 0 0 10px #ff4b4b;
    }

    /* Optional: add hover effect */
    button[data-testid="baseButton-headerNoPadding"]:hover {
        box-shadow: 0 0 20px #ff0000;
        transform: scale(1.1);
        transition: 0.2s ease-in-out;
    }
    </style>
    """,
    unsafe_allow_html=True
)


    # --- Title ---
    st.markdown("<h1 class='main-title'>✈️ Aircraft Component Failure Prediction Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>An AI-powered logistic regression system for predicting aircraft component failure probabilities.</p>", unsafe_allow_html=True)

    # --- Sidebar Inputs ---
    st.sidebar.header("🧩 Component Input Parameters")

    features = {
        "Temperature (°C)": st.sidebar.slider("Temperature (°C)", 200, 900, 700),
        "Pressure (psi)": st.sidebar.slider("Pressure (psi)", 100, 400, 300),
        "Vibration (g)": st.sidebar.slider("Vibration (g)", 0.0, 1.0, 0.3),
        "Oil Quality (%)": st.sidebar.slider("Oil Quality (%)", 0, 100, 80),
        "Flight Hours": st.sidebar.slider("Flight Hours", 0, 5000, 1500),
        "Ambient Temp (°C)": st.sidebar.slider("Ambient Temp (°C)", -20, 50, 25),
        "Altitude (ft)": st.sidebar.slider("Altitude (ft)", 0, 15000, 10000),
        "Last Service Cycles": st.sidebar.slider("Last Service Cycles", 0, 500, 150)
    }

    X_input = pd.DataFrame([features], columns=X.columns)
    X_scaled = scaler.transform(X_input)
    prob = predict_prob(X_scaled, theta)[0]

    # --- Tabs ---
    tabs = st.tabs(["🧠 Prediction", "📊 Feature Insights", "📈 Data Visualization", "ℹ️ About Project"])

    # --- Prediction Tab ---
    with tabs[0]:
        st.subheader("Prediction Results")
        st.metric("Failure Probability", f"{prob * 100:.2f}%")

        progress_val = int(prob * 100)
        st.progress(progress_val)

        if prob >= 0.6 and prob <0.9:
            st.error("⚠️ **High Risk:** Component likely to FAIL soon!")
            st.markdown("🧯 **Recommendation:** Immediate inspection and maintenance required.")
        elif prob>=0.9:
            st.error("⚠️ **EMERGENCY!!!** Aircraft in danger")
            st.markdown("🧯 **Recommendation:** Emergency landing NOW.")
        else:
            st.success("✅ **Low Risk:** Component operating within safe parameters.")
            st.markdown("🛫 **Recommendation:** Continue normal operation, monitor parameters regularly.")

        st.divider()
        # Download prediction report
        report_data = {**features, "Failure Probability": prob}
        report_df = pd.DataFrame([report_data])
        csv = report_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Prediction Report",
            data=csv,
            file_name="component_prediction_report.csv",
            mime="text/csv"
        )

    # --- Feature Insights Tab ---
    with tabs[1]:
        st.subheader("Feature Importance Analysis")
        feature_names = list(features.keys())
        coefficients = theta[1:]
        importance = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": coefficients,
            "AbsValue": np.abs(coefficients)
        }).sort_values(by="AbsValue", ascending=False)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x="Feature", y="Coefficient", data=importance, palette="coolwarm", ax=ax)
        plt.xticks(rotation=45)
        plt.title("Feature Influence on Failure Probability")
        st.pyplot(fig)

    # --- Visualization Tab ---
    with tabs[2]:
        st.subheader("Feature Distributions (Failure vs Normal)")
        feature_cols = ["Temperature (°C)", "Pressure (psi)", "Vibration (g)", "Oil Quality (%)"]

        for f in feature_cols:
            with st.expander(f"📈 View {f} Distribution"):
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.kdeplot(df[df["Failure Probability"] < 0.5][f], label="Normal", fill=True, alpha=0.5)
                sns.kdeplot(df[df["Failure Probability"] >= 0.5][f], label="Fail", fill=True, alpha=0.5)
                ax.axvline(X_input[f].values[0], color="red", linestyle="--", label="Current Input")
                plt.title(f"{f} Distribution")
                plt.legend()
                st.pyplot(fig)

    # --- About Tab ---
    with tabs[3]:
        st.subheader("About This Project")
        st.markdown("""
            **Project Title:** Aircraft Component Failure Prediction  
            **Developer:** *Muktar Sanusi* ✈️  
            **Institution:** Moscow Aviation Institute (MAI)  
            **Description:**  
            This interactive machine learning dashboard applies logistic regression to predict the probability of component failure in aircraft systems.  
            The model was trained on simulated aviation sensor data (temperature, pressure, vibration, etc.), providing real-time predictive analytics for aircraft maintenance.  
    
            **Key Features:**  
            - Real-time failure prediction  
            - Scalable and interpretable logistic regression model  
            - Visual analysis of feature contributions  
            - Interactive Streamlit dashboard  
            - Downloadable prediction reports  
        """)

    # --- Footer ---
    st.markdown("""
        <div class="footer">
            © 2025 Muktar Sanusi | Aircraft Engineering & AI/ML Engineer/Enthusiast | Moscow Aviation Institute (MAI)
        </div>
    """, unsafe_allow_html=True)
app()
