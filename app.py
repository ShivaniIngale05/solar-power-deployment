import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------- 1. PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Solar Power Prediction",
    page_icon="☀️",
    layout="centered"
)

# ---------------- 2. DARK AESTHETIC THEME  ----------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0E1117;
        color: #EAEAEA;
    }
    h1, h2, h3 {
        color: #F5C518;
        text-align: center;
    }
    .stButton > button {
        background-color: #F5C518;
        color: #0E1117;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-weight: 600;
    }
    .stSlider > label {
        color: #EAEAEA;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- 3. HEADER & IMAGE ----------------
st.title("Solar Power Prediction")
st.caption("Advanced Machine Learning Forecasting System")

# Use the solar image from your repository
try:
    st.image("solar.png", use_container_width=True)
except:
    # Fallback image if solar.png is missing
    st.image("https://images.unsplash.com/photo-1508514177221-188b1cf16e9d?auto=format&fit=crop&w=800&q=80")

# ---------------- 4. LOAD DATA & MODEL ----------------
try:
    # Load your specific trained Gradient Boosting model
    model = pickle.load(open("final_gb_model.pkl", "rb"))
    # Load your CSV to get feature ranges
    df = pd.read_csv("final_model_data.csv")
except FileNotFoundError:
    st.error("⚠️ Error: 'final_gb_model.pkl' or 'final_model_data.csv' not found in GitHub!")
    st.stop()

# ---------------- 5. UNIT MAPPING ----------------
UNIT_MAP = {
    "humidity": "%",
    "wind_direction": "°",
    "wind_speed": "m/s",
    "pressure": "hPa",
    "distance_to_solar_noon": "radians",
    "sky_cover": "scale (0–4)"
}

# ---------------- 6. INPUT SLIDERS ----------------
st.subheader("Environmental Parameters")

# Get feature names and remove target if present
features = df.columns.tolist()
if 'power-generated' in features:
    features.remove('power-generated')

user_input = []

# Create a clean two-column layout
col1, col2 = st.columns(2)

for i, feature in enumerate(features):
    # Get units from the map
    unit = next((UNIT_MAP[k] for k in UNIT_MAP if k in feature.lower()), "")
    label = f"{feature} {f'({unit})' if unit else ''}"
    
    # Alternate sliders between columns
    with col1 if i % 2 == 0 else col2:
        val = st.slider(
            label,
            min_value=float(df[feature].min()),
            max_value=float(df[feature].max()),
            value=float(df[feature].mean())
        )
        user_input.append(val)

# ---------------- 7. PREDICTION LOGIC ----------------
if st.button("Predict Power Generation"):
    # Convert input to DataFrame
    input_df = pd.DataFrame([user_input], columns=features)
    
    # Get prediction from your model
    prediction = model.predict(input_df)[0]
    
    # NEW: Ensure the result is never negative
    final_result = max(0, prediction)
    
    st.markdown("---")
    st.success(f"### 🔋 Predicted Solar Power: **{final_result:.2f} kW**")
