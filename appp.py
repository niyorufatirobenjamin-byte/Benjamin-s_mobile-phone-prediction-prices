# app.py
import streamlit as st
import pandas as pd
import pickle

# Load model, scaler, and encoders
model = pickle.load(open("phone_model.sav", "rb"))
scaler = pickle.load(open("scaler.sav", "rb"))
label_encoders = pickle.load(open("encoders.sav", "rb"))

st.title("Mobile Phone Price Prediction")

# Input fields
Brand = st.text_input("Brand (text)")
Battery_capacity = st.number_input("Battery capacity (mAh)", min_value=500, max_value=20000, value=4000)
Screen_size = st.number_input("Screen size (inches)", min_value=3.0, max_value=10.0, value=6.5)
Touchscreen = st.selectbox("Touchscreen", ["Yes", "No"])
Processor = st.number_input("Processor (numeric code)", min_value=1, max_value=20, value=8)
RAM = st.number_input("RAM (MB)", min_value=512, max_value=64000, value=12000)
Internal_storage = st.number_input("Internal storage (GB)", min_value=8, max_value=2048, value=256)
Rear_camera = st.number_input("Rear camera (MP)", min_value=1, max_value=200, value=48)
Operating_system = st.selectbox("Operating system", ["Android", "iOS", "Other"])
Number_of_SIMs = st.number_input("Number of SIMs", min_value=1, max_value=3, value=2)
ThreeG = st.selectbox("3G Support", ["Yes", "No"])
FourG = st.selectbox("4G/LTE Support", ["Yes", "No"])

if st.button("Predict Price"):
    try:
        # Create DataFrame from user input
        new_phone = pd.DataFrame([{
            "Brand": Brand,
            "Battery capacity (mAh)": Battery_capacity,
            "Screen size (inches)": Screen_size,
            "Touchscreen": Touchscreen,
            "Processor": Processor,
            "RAM (MB)": RAM,
            "Internal storage (GB)": Internal_storage,
            "Rear camera": Rear_camera,
            "Operating system": Operating_system,
            "Number of SIMs": Number_of_SIMs,
            "3G": ThreeG,
            "4G/ LTE": FourG
        }])

        # Encode categorical columns
        for col, le in label_encoders.items():
            new_phone[col] = le.transform(new_phone[col])

        # Scale features
        new_phone_scaled = scaler.transform(new_phone)

        # Predict price
        price = model.predict(new_phone_scaled)
        st.success(f"The predicted price is: RWF {price[0]:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")