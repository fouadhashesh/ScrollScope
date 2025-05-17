import streamlit as st
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import shap

# --- Page configuration ---
st.set_page_config(page_title="Screen Addiction Prediction", layout="wide")

@st.cache(allow_output_mutation=True)
def load_artifacts():
    """
    Load and cache scaler and model, create a SHAP KernelExplainer using scaled background.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Load artifacts
    scaler = joblib.load(os.path.join(base_dir, "scaler.pkl"))
    model = joblib.load(os.path.join(base_dir, "NN_Classification_model.pkl"))
    # Prepare background in scaled feature space
    background = scaler.transform(np.zeros((1, 10)))
    # Use KernelExplainer on the model.predict method (returns probability of class_1 = not addicted)
    explainer = shap.KernelExplainer(model.predict, background)
    return scaler, model, explainer

scaler, model, explainer = load_artifacts()

# Initialize history storage
if "history" not in st.session_state:
    st.session_state.history = []

# --- Sidebar: Inputs & Settings ---
st.sidebar.header("User Inputs & Settings")
inputs = {
    "Daily Screen Time (hrs)": st.sidebar.slider("Daily Screen Time", 0.0, 24.0, 2.0, 0.1),
    "App Sessions per Day": st.sidebar.number_input("App Sessions", 0, 200, 20, 1),
    "Social Media Usage (hrs/day)": st.sidebar.slider("Social Media Usage", 0.0, 24.0, 1.0, 0.1),
    "Gaming Time (hrs/day)": st.sidebar.slider("Gaming Time", 0.0, 24.0, 0.5, 0.1),
    "Notifications per Day": st.sidebar.number_input("Notifications", 0, 1000, 50, 1),
    "Night Usage (hrs)": st.sidebar.slider("Night Usage", 0.0, 12.0, 0.5, 0.1),
    "Age": st.sidebar.number_input("Age", 1, 120, 25, 1),
    "Work/Study Hours (hrs/day)": st.sidebar.slider("Work/Study Hours", 0.0, 24.0, 8.0, 0.1),
    "Stress Level (1=Low,10=High)": st.sidebar.slider("Stress Level", 1, 10, 5),
    "Apps Installed": st.sidebar.number_input("Apps Installed", 0, 500, 50, 1),
}
threshold = st.sidebar.slider("Addiction Threshold (%)", 0, 100, 50, 1) / 100.0

# Helper to assemble and scale features
def get_scaled_features():
    arr = np.array([
        inputs["Daily Screen Time (hrs)"],
        inputs["App Sessions per Day"],
        inputs["Social Media Usage (hrs/day)"],
        inputs["Gaming Time (hrs/day)"],
        inputs["Notifications per Day"],
        inputs["Night Usage (hrs)"],
        inputs["Age"],
        inputs["Work/Study Hours (hrs/day)"],
        inputs["Stress Level (1=Low,10=High)"],
        inputs["Apps Installed"],
    ]).reshape(1, -1)
    return scaler.transform(arr)

# --- Main Page ---
st.title("📊 Screen Addiction Prediction")
st.markdown("Enter data via the sidebar and click Predict. Adjust threshold as needed.")

# Display inputs and threshold
col1, col2 = st.columns(2)
with col1:
    st.subheader("Current Inputs")
    st.write(inputs)
with col2:
    st.subheader("Threshold")
    st.write(f"{threshold * 100:.0f}%")

# --- Prediction Logic ---
if st.button("Predict Addiction"):
    X_scaled = get_scaled_features()
    # model.predict returns probability of class_1 = not addicted
    proba_not_addicted = model.predict(X_scaled)[0][0]
    # Convert to addiction probability
    proba_addicted = 1 - proba_not_addicted
    # Label: 0 = addicted, 1 = not addicted
    label = 0 if proba_addicted >= threshold else 1

    # Display result
    left, right = st.columns([1, 2])
    with left:
        if label == 0:
            st.error("🔴 Addicted")
        else:
            st.success("🟢 Not Addicted")
    with right:
        st.metric(label="Addiction Probability", value=f"{proba_addicted:.2%}")

    # Log history
    st.session_state.history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **inputs,
        "probability": float(proba_addicted),
        "prediction": "Addicted" if label == 0 else "Not Addicted",
        "threshold": threshold,
    })

# --- SHAP Explainability ---
if st.button("Explain Prediction"):
    if not st.session_state.history:
        st.warning("Make a prediction first!")
    else:
        # Compute SHAP values and flatten to 1D
        X_scaled = get_scaled_features()
        shap_vals = explainer.shap_values(X_scaled)
        shap_array = np.array(shap_vals)
        shap_flat = shap_array.flatten()
        # Build DataFrame for chart
        df_shap = pd.DataFrame({
            "feature": list(inputs.keys()),
            "shap_value": shap_flat
        }).set_index("feature")
        st.subheader("Feature Contributions (SHAP)")
        st.bar_chart(df_shap)

# --- History & Export ---
if st.session_state.history:
    st.markdown("---")
    st.subheader("Prediction History")
    df_hist = pd.DataFrame(st.session_state.history)
    st.dataframe(df_hist)
    csv = df_hist.to_csv(index=False)
    st.download_button("Download History as CSV", data=csv, file_name="history.csv", mime="text/csv")
    if st.button("Clear History"):
        st.session_state.history = []
        st.experimental_rerun()

# Footer
st.markdown("---")
st.caption("Model predictions with adjustable threshold and SHAP explanations.")
