import math
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

# Load historical data
data = pd.read_csv('modelData.csv')

# Preprocess the dataset
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%y')
data['year'] = data['Date'].dt.year
data['dayofweek'] = data['Date'].dt.dayofweek  # Monday=0, Sunday=6
data['weekofyear'] = data['Date'].dt.isocalendar().week
data['is_weekend'] = (data['dayofweek'] >= 5).astype(int)

# Encode categorical features
le = LabelEncoder()
data['conditions'] = le.fit_transform(data['conditions'])

# Load trained XGBoost model
xgb_model = joblib.load('xgb_model2.pkl')  # Load the trained model

# Streamlit UI
st.image("RisingTide.jpg", use_container_width=False)
st.markdown("------------")

# Center the title (HTML and CSS)
st.markdown("<h1 style='text-align: center;'>RTCW Daily Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Please enter appropriate weather information and click Predict to plan your day!</p>", unsafe_allow_html=True)
prev_day_count = st.number_input("Please Enter Previous day's car count (Can be approximate)", 0, 1000, 500)
st.markdown("-------------")

st.sidebar.header("Enter Weather Data")
temp = st.sidebar.number_input("Temperature (°F)", value=75)
humidity = st.sidebar.number_input("Humidity (%)", value=50)
precip = st.sidebar.number_input("Precipitation (inches)", value=0.1)
precipcover = st.sidebar.slider("Chance of Rain (%)", 0, 100, 10)
cloudcover = st.sidebar.slider("Cloud Cover (%)", 0, 100, 50)
uvindex = st.sidebar.number_input("UV Index", value=5)
dayofweek = st.sidebar.selectbox("Day of the Week", list(range(7)), format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x])
conditions = st.sidebar.selectbox("Weather Conditions", le.classes_)
AQI = st.sidebar.slider("Air Quality Index", 0, 100, 30)

# Encode conditions using LabelEncoder
conditions_encoded = le.transform([conditions])[0]

# Approximate previous day rain & counts
prev_day_rain = precip  # Use today's precipitation (Approx)
prev_day_count = prev_day_count # User enters car count for prev day

# Create input to match trained model
input_data = pd.DataFrame({
    'temp': [temp],
    'humidity': [humidity],
    'precip': [precip],
    'precipcover': [precipcover],
    'cloudcover': [cloudcover],
    'uvindex': [uvindex],
    'conditions': [conditions_encoded],
    'AQI': [AQI],
    'year': [2024],  # Placeholder
    'dayofweek': [dayofweek],
    'weekofyear': [10],  # Placeholder
    'is_weekend': [1 if dayofweek >= 5 else 0],
    'prev_day_rain': [prev_day_rain],
    'prev_day_count': [prev_day_count],
    'rolling_rain_2': [precip],  # Approximate
    'rolling_rain_3': [precip],  # Approximate
    'rolling_rain_7': [precip]   # Approximate
})
# Calculate FS count from car count prediction (Weekends usually busier)
if dayofweek < 4:  # Monday–Friday (0–4)
    FSmultiplier = 0.11
else:  # Saturday–Sunday (5,6)
    FSmultiplier = 0.18

multiplier = 1.0 # Default
# Penalize rainy days for better accuracy
if precipcover > 30 or (conditions == "Rain, Partially Cloudy" or conditions == "Rain"):
    multiplier =  0.4

# Make prediction
if st.sidebar.button("Predict"):
    prediction = xgb_model.predict(input_data)[0]
    members = prediction*0.60
    members = members * multiplier
    conversion = members*0.10
    st.subheader(f"Car Wash Count for the day (Retail & Members): {int(prediction)} cars")
    st.subheader(f"Predicted FS washes: {int(prediction)*FSmultiplier :.0f} cars")
    st.markdown("------------------------")
    st.subheader(f"Predicted Potential Members: {members:.0f} cars")
    st.subheader(f"Conversion Goal: {conversion:.0f} new members")
    st.markdown("-----------------------")

    # Greeter split
    st.header("Shift Split for Predicted New Members")
    leftover = conversion % 2

    # Display greeter result
    st.subheader("Recommended Distribution")
    st.write(f"Opening Greeter: **{math.ceil(conversion // 2)}** new members")
    st.write(f"Closing Greeter: **{math.ceil(conversion // 2)}** new members")
    st.write(f"Sales Supervisor/Manager: **{math.ceil(leftover)}** new members")