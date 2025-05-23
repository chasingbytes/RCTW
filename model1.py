import math
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

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

st.set_page_config(page_title="Sales Predictor", page_icon="🚘")
# path settings
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / "main.css"
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
# Streamlit UI
left_co, cent_co, last_co = st.columns(3)
with cent_co:
    st.image("RisingTide.jpg", use_container_width=False)
st.markdown("------------")

# Center the title (HTML and CSS)
st.markdown("<h1 style='text-align: center;'>RTCW Daily Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Please enter appropriate weather information on the sidebar and click Predict to plan your day!</p>", unsafe_allow_html=True)
# prev_day_count = st.number_input("Please Enter Previous day's car count (Can be approximate)", 0, 1000, 500)
st.markdown("-------------")

st.sidebar.header("Enter Weather Data")
prev_day_count = st.sidebar.number_input("Enter Previous day's car count (Find this on Vehicle Performance)", 0, 1000, 500)
temp = st.sidebar.number_input("Temperature (°F)", value=75)
humidity = st.sidebar.number_input("Humidity (%)", value=50)
precip = st.sidebar.number_input("Precipitation (inches)", value=0.1)
precipcover = st.sidebar.slider("Chance of Rain (%)", 0, 100, 10)
uvindex = st.sidebar.number_input("UV Index", value=5)
dayofweek = st.sidebar.selectbox("Day of the Week", list(range(7)), format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x])
conditions = st.sidebar.selectbox("Current Weather Conditions", le.classes_)
AQI = st.sidebar.slider("Air Quality Index", 0, 100, 30)

# Encode conditions using LabelEncoder
conditions_encoded = le.transform([conditions])[0]

# Approximate previous day rain & counts
prev_day_rain = precip  # Use today's precipitation (Approx)
prev_day_count = prev_day_count # User enters car count for prev day
cloudcover = precip

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
    FSmultiplier = 0.09
else:  # Saturday–Sunday (5,6)
    FSmultiplier = 0.17

multiplier = 1.0 # Default
# Penalize rainy days for better accuracy
if precipcover > 40 and (conditions == "Rain, Partially Cloudy" or conditions == "Rain"):
    multiplier =  0.4
if precipcover > 20 and (conditions == "Partially Cloudy" or conditions == "Overcast" or conditions == "Rain, Overcast"):
    multiplier = 0.625
if conditions == "Rain" or conditions == "Rain, Overcast" or conditions == "Rain, Partially Cloudy":
    multiplier = 0.4
if conditions == "Overcast" or conditions == "Partially Cloudy":
    multiplier = 0.7

# Make prediction
if st.sidebar.button("Predict"):
    prediction = xgb_model.predict(input_data)[0] * multiplier
    members = prediction * 0.45
    members = members * multiplier
    conversion = members * 0.10
    st.subheader(f"Car Wash Count for the day (Retail & Members): {int(prediction)} cars")
    st.subheader(f"Predicted FS washes: {int(prediction)*FSmultiplier :.0f} cars")
    st.markdown("------------------------")
    st.subheader(f":blue_car: Predicted Potential Members: {members:.0f} cars :blue_car:")
    st.subheader(f":zap: Conversion Goal: {conversion:.0f} new members :zap:")
    st.markdown("-----------------------")
    st.subheader(f":star: Greeters should aim for {(conversion / 11):.0f} new members per hour :star:")
    st.markdown("-----------------------")

    # Greeter split
    st.header("Shift Split for Predicted New Members")
    greeter = math.ceil(conversion // 2)
    leftover = conversion % 2

    # Display greeter result
    st.subheader("Recommended Distribution")
    st.write(f"Opening Greeter Team: **{greeter}** new members")
    st.write(f"Closing Greeter Team: **{greeter}** new members")
    st.write(f"Sales Supervisor/Manager: **{math.ceil(leftover)}** new members")
