"""
Food Delivery Time Prediction — Streamlit App
Author: Abdul Rehman
Model: XGBoost Regressor (Test R2 ~0.83, Test RMSE ~3.9 min)

This single file is both the "frontend" (Streamlit UI) and "backend"
(loading the trained model/scaler + running inference) — no separate
API server is needed, Streamlit handles both.
"""

import numpy as np
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path

# --------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------
# Constants — MUST mirror the notebook's feature engineering exactly
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.json"
SCALER_PATH = BASE_DIR / "standard_scaler.pkl"

TRAFFIC_MAPPING = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}

TYPE_OF_ORDER_OPTIONS = ["Buffet", "Drinks", "Meal", "Snack"]
TYPE_OF_VEHICLE_OPTIONS = ["bicycle", "electric_scooter", "motorcycle", "scooter"]
CITY_OPTIONS = ["Metropolitian", "Semi-Urban", "Urban"]
WEATHER_OPTIONS = ["Cloudy", "Fog", "Sandstorms", "Stormy", "Sunny", "Windy"]
FESTIVAL_OPTIONS = ["No", "Yes"]

# Exact column order the scaler/model were fitted on (from X_train.columns)
FEATURE_COLUMNS = [
    "Delivery_person_Age", "Delivery_person_Ratings", "distance_km",
    "order_hour", "Road_traffic_density", "Vehicle_condition",
    "multiple_deliveries", "Type_of_order_Drinks", "Type_of_order_Meal",
    "Type_of_order_Snack", "Type_of_vehicle_electric_scooter",
    "Type_of_vehicle_motorcycle", "Type_of_vehicle_scooter", "Festival_Yes",
    "City_Semi-Urban", "City_Urban", "Weatherconditions_Fog",
    "Weatherconditions_Sandstorms", "Weatherconditions_Stormy",
    "Weatherconditions_Sunny", "Weatherconditions_Windy",
]

SOCIAL_LINKS = {
    "GitHub": "https://github.com/datawithabdulrehman",
    "Kaggle": "https://www.kaggle.com/datawithabxrehman",
    "LinkedIn": "https://www.linkedin.com/in/datawithabdulrehman",
}

# --------------------------------------------------------------------------
# Styling — clean, white, professional
# --------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

    .stApp { background-color: #F7F8FA; }

    #MainMenu, footer, header { visibility: hidden; }

    .block-container { padding-top: 2rem; padding-bottom: 1rem; max-width: 1100px; }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #101828 0%, #1D2939 100%);
        border-radius: 20px;
        padding: 2.4rem 2.6rem;
        margin-bottom: 1.8rem;
        color: white;
        box-shadow: 0 10px 30px rgba(16, 24, 40, 0.15);
    }
    .hero h1 {
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
        color: #FFFFFF;
        letter-spacing: -0.02em;
    }
    .hero p {
        font-size: 1.02rem;
        color: #D0D5DD;
        margin: 0;
        font-weight: 400;
    }
    .hero .badge {
        display: inline-block;
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.18);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        color: #EAECF0;
        margin-top: 1rem;
        margin-right: 8px;
    }

    /* Section cards */
    .section-card {
        background: #FFFFFF;
        border: 1px solid #EAECF0;
        border-radius: 16px;
        padding: 1.5rem 1.6rem 0.6rem 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.04);
    }
    .section-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: #101828;
        margin-bottom: 0.9rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Predict button */
    div.stButton > button {
        background: #101828;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.15s ease;
    }
    div.stButton > button:hover {
        background: #344054;
        color: white;
        transform: translateY(-1px);
    }

    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #ECFDF3 0%, #F7F8FA 100%);
        border: 1px solid #A6F4C5;
        border-radius: 18px;
        padding: 2rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-minutes {
        font-size: 3.2rem;
        font-weight: 800;
        color: #067647;
        line-height: 1;
        margin: 0.3rem 0;
    }
    .result-label {
        color: #475467;
        font-size: 0.95rem;
        font-weight: 500;
    }

    /* Footer */
    .footer {
        margin-top: 3rem;
        padding-top: 1.8rem;
        border-top: 1px solid #EAECF0;
        text-align: center;
    }
    .footer-icons { display: flex; justify-content: center; gap: 16px; margin-bottom: 0.9rem; }
    .footer-icons a {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 42px;
        height: 42px;
        border-radius: 50%;
        background: #FFFFFF;
        border: 1px solid #EAECF0;
        transition: all 0.15s ease;
        box-shadow: 0 1px 2px rgba(16,24,40,0.05);
    }
    .footer-icons a:hover { background: #101828; transform: translateY(-2px); }
    .footer-icons a:hover svg { fill: #FFFFFF; }
    .footer-icons svg { width: 20px; height: 20px; fill: #344054; transition: fill 0.15s ease; }
    .footer-text { color: #98A2B3; font-size: 0.85rem; }

    div[data-testid="stMetricValue"] { color: #101828; }
    label { font-weight: 500 !important; color: #344054 !important; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Load model + scaler (cached so it only happens once per session)
# --------------------------------------------------------------------------
import xgboost as xgb
import joblib

@st.cache_resource(show_spinner=False)
def load_artifacts():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        return None, None
    
    # Naya aur mehfooz tareeqa
    model = xgb.XGBRegressor() # Agar classification hai toh XGBClassifier() likhein
    model.load_model(str(MODEL_PATH))
    
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def haversine_km(lat1, lon1, lat2, lon2):
    """Same great-circle distance formula used in the training notebook."""
    r = 6371  # earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * r


def build_feature_row(inputs: dict) -> pd.DataFrame:
    """Turns the raw form inputs into the exact 21-column row the scaler/model expect."""
    row = {col: 0 for col in FEATURE_COLUMNS}

    row["Delivery_person_Age"] = inputs["age"]
    row["Delivery_person_Ratings"] = inputs["ratings"]
    row["distance_km"] = inputs["distance_km"]
    row["order_hour"] = inputs["order_hour"]
    row["Road_traffic_density"] = TRAFFIC_MAPPING[inputs["traffic"]]
    row["Vehicle_condition"] = inputs["vehicle_condition"]
    row["multiple_deliveries"] = inputs["multiple_deliveries"]

    # One-hot: Type_of_order (base category dropped: Buffet)
    if inputs["order_type"] != "Buffet":
        row[f"Type_of_order_{inputs['order_type']}"] = 1

    # One-hot: Type_of_vehicle (base category dropped: bicycle)
    if inputs["vehicle_type"] != "bicycle":
        row[f"Type_of_vehicle_{inputs['vehicle_type']}"] = 1

    # One-hot: Festival (base category dropped: No)
    if inputs["festival"] == "Yes":
        row["Festival_Yes"] = 1

    # One-hot: City (base category dropped: Metropolitian)
    if inputs["city"] != "Metropolitian":
        row[f"City_{inputs['city']}"] = 1

    # One-hot: Weatherconditions (base category dropped: Cloudy)
    if inputs["weather"] != "Cloudy":
        row[f"Weatherconditions_{inputs['weather']}"] = 1

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_minutes(model, scaler, feature_row: pd.DataFrame) -> float:
    scaled = scaler.transform(feature_row)
    pred = model.predict(scaled)
    return float(pred[0])


# --------------------------------------------------------------------------
# Hero header
# --------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🛵 Food Delivery Time Predictor</h1>
    <p>Machine learning model that estimates delivery time (in minutes) from order,
    rider and route conditions — powered by a tuned XGBoost regressor.</p>
    <span class="badge">📊 Test R² ≈ 0.83</span>
    <span class="badge">⏱️ Test RMSE ≈ 3.9 min</span>
    <span class="badge">🌲 XGBoost Regressor</span>
</div>
""", unsafe_allow_html=True)

model, scaler = load_artifacts()

if model is None or scaler is None:
    st.error(
        "Model files not found. Make sure `xgb_best.pkl` and `standard_scaler.pkl` "
        "are in the same folder as `app.py`."
    )
    st.stop()

# --------------------------------------------------------------------------
# Input form
# --------------------------------------------------------------------------
with st.form("prediction_form"):

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧑‍🍳 Delivery Person</div>', unsafe_allow_html=True)
        age = st.slider("Age", min_value=15, max_value=50, value=29)
        ratings = st.slider("Rating", min_value=1.0, max_value=5.0, value=4.6, step=0.1)
        vehicle_type = st.selectbox("Vehicle Type", TYPE_OF_VEHICLE_OPTIONS, index=2,
                                     format_func=lambda x: x.replace("_", " ").title())
        vehicle_condition = st.select_slider(
            "Vehicle Condition (0 = worst, 3 = best)", options=[0, 1, 2, 3], value=1
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📦 Order Details</div>', unsafe_allow_html=True)
        order_type = st.selectbox("Type of Order", TYPE_OF_ORDER_OPTIONS, index=3)
        order_hour = st.slider("Order Hour (24h)", min_value=0, max_value=23, value=13)
        multiple_deliveries = st.select_slider(
            "Multiple Deliveries (same trip)", options=[0, 1, 2, 3], value=0
        )
        festival = st.radio("Festival Day?", FESTIVAL_OPTIONS, index=0, horizontal=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📍 Route (Restaurant → Customer)</div>', unsafe_allow_html=True)
        rc1, rc2 = st.columns(2)
        with rc1:
            rest_lat = st.number_input("Restaurant Latitude", value=22.745049, format="%.6f")
            rest_lon = st.number_input("Restaurant Longitude", value=75.892471, format="%.6f")
        with rc2:
            del_lat = st.number_input("Delivery Latitude", value=22.765049, format="%.6f")
            del_lon = st.number_input("Delivery Longitude", value=75.912471, format="%.6f")
        st.caption("Distance is calculated automatically using the haversine formula (same as training).")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🌦️ Conditions</div>', unsafe_allow_html=True)
        weather = st.selectbox("Weather", WEATHER_OPTIONS, index=4)
        traffic = st.selectbox("Road Traffic Density", list(TRAFFIC_MAPPING.keys()), index=1)
        city = st.selectbox("City Type", CITY_OPTIONS, index=2)
        st.markdown('</div>', unsafe_allow_html=True)

    submitted = st.form_submit_button("🔮 Predict Delivery Time")

if submitted:
    distance_km = haversine_km(rest_lat, rest_lon, del_lat, del_lon)

    if distance_km > 100:
        st.warning(
            f"Calculated distance is {distance_km:.1f} km, which is outside the model's "
            "reliable training range (< 100 km). Double check the coordinates — the "
            "prediction below may not be accurate."
        )

    inputs = dict(
        age=age, ratings=ratings, distance_km=distance_km, order_hour=order_hour,
        traffic=traffic, vehicle_condition=vehicle_condition,
        multiple_deliveries=multiple_deliveries, order_type=order_type,
        vehicle_type=vehicle_type, festival=festival, city=city, weather=weather,
    )

    feature_row = build_feature_row(inputs)
    predicted_minutes = predict_minutes(model, scaler, feature_row)
    predicted_minutes = max(predicted_minutes, 0)

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Estimated Delivery Time</div>
        <div class="result-minutes">{predicted_minutes:.0f} min</div>
        <div class="result-label">Distance: {distance_km:.2f} km &nbsp;•&nbsp; Model: XGBoost Regressor</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("See the exact feature values sent to the model"):
        st.dataframe(feature_row.T.rename(columns={0: "value"}), use_container_width=True)

# --------------------------------------------------------------------------
# Footer with social icons
# --------------------------------------------------------------------------
st.markdown(f"""
<div class="footer">
    <div class="footer-icons">
        <a href="{SOCIAL_LINKS['GitHub']}" target="_blank" title="GitHub">
            <svg viewBox="0 0 24 24"><path d="M12 .5C5.73.5.98 5.24.98 11.52c0 5.02 3.26 9.28 7.77 10.78.57.1.78-.25.78-.55 0-.27-.01-1-.02-1.96-3.16.69-3.83-1.52-3.83-1.52-.52-1.31-1.26-1.66-1.26-1.66-1.03-.7.08-.69.08-.69 1.14.08 1.74 1.17 1.74 1.17 1.01 1.74 2.65 1.24 3.3.95.1-.73.4-1.24.72-1.53-2.52-.29-5.17-1.26-5.17-5.6 0-1.24.44-2.25 1.17-3.04-.12-.29-.51-1.45.11-3.02 0 0 .96-.31 3.14 1.16a10.9 10.9 0 0 1 5.72 0c2.18-1.47 3.14-1.16 3.14-1.16.62 1.57.23 2.73.11 3.02.73.79 1.17 1.8 1.17 3.04 0 4.35-2.65 5.31-5.18 5.59.41.35.77 1.04.77 2.1 0 1.52-.01 2.74-.01 3.11 0 .3.2.66.79.55A11.03 11.03 0 0 0 23.02 11.5C23.02 5.24 18.27.5 12 .5z"/></svg>
        </a>
        <a href="{SOCIAL_LINKS['Kaggle']}" target="_blank" title="Kaggle">
            <svg viewBox="0 0 24 24"><path d="M18.83 21.06h-3.4c-.13 0-.25-.06-.33-.16l-4.99-6.22-1.4 1.34v4.71c0 .18-.15.33-.33.33H5.62a.33.33 0 0 1-.33-.33V2.94c0-.18.15-.33.33-.33h2.76c.18 0 .33.15.33.33v11.6l5.94-6.03a.5.5 0 0 1 .36-.15h3.53c.27 0 .41.33.22.53l-6.2 6.13 6.63 8.5a.33.33 0 0 1-.26.54z"/></svg>
        </a>
        <a href="{SOCIAL_LINKS['LinkedIn']}" target="_blank" title="LinkedIn">
            <svg viewBox="0 0 24 24"><path d="M20.45 20.45h-3.55v-5.57c0-1.33-.02-3.04-1.85-3.04-1.86 0-2.14 1.45-2.14 2.95v5.66H9.36V9h3.41v1.56h.05c.47-.9 1.63-1.85 3.36-1.85 3.6 0 4.27 2.37 4.27 5.45v6.29zM5.34 7.43a2.06 2.06 0 1 1 0-4.12 2.06 2.06 0 0 1 0 4.12zM7.12 20.45H3.56V9h3.56v11.45z"/></svg>
        </a>
    </div>
    <div class="footer-text">Built by Abdul Rehman &nbsp;•&nbsp; BS Data Science &nbsp;•&nbsp; Powered by Streamlit + XGBoost</div>
</div>
""", unsafe_allow_html=True)
