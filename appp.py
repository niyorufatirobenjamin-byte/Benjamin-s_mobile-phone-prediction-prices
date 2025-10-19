import streamlit as st
import pandas as pd
import pickle

# Add your name / designed by line at the top
st.markdown("### Mobile Phone Price Prediction")
st.markdown("*Designed by Benjamin NIYORUFATIRO*")
st.write("---")  # optional horizontal line for separation

# Load saved model, scaler, and label encoders
model = pickle.load(open("phone_model.sav", "rb"))
scaler = pickle.load(open("scaler.sav", "rb"))
label_encoders = pickle.load(open("encoders.sav", "rb"))

# Dropdowns for categorical features using LabelEncoder classes
Brand = st.selectbox("Brand", label_encoders['Brand'].classes_)
Touchscreen = st.selectbox("Touchscreen", label_encoders['Touchscreen'].classes_)
Operating_system = st.selectbox("Operating system", label_encoders['Operating system'].classes_)
ThreeG = st.selectbox("3G Support", label_encoders['3G'].classes_)
FourG = st.selectbox("4G/LTE Support", label_encoders['4G/ LTE'].classes_)

# Numeric inputs
Battery_capacity = st.number_input("Battery capacity (mAh)", min_value=500, max_value=20000, value=4000)
Screen_size = st.number_input("Screen size (inches)", min_value=3.0, max_value=10.0, value=6.5)
Processor = st.number_input("Processor (numeric code)", min_value=1, max_value=20, value=8)
RAM = st.number_input("RAM (MB)", min_value=512, max_value=64000, value=12000)
Internal_storage = st.number_input("Internal storage (GB)", min_value=8, max_value=2048, value=256)
Rear_camera = st.number_input("Rear camera (MP)", min_value=1, max_value=200, value=48)
Number_of_SIMs = st.number_input("Number of SIMs", min_value=1, max_value=3, value=2)

if st.button("Predict Price"):
    try:
        # Create DataFrame from inputs
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

        # Encode categorical columns using saved LabelEncoders
        for col, le in label_encoders.items():
            new_phone[col] = le.transform(new_phone[col])

        # Scale numeric features
        new_phone_scaled = scaler.transform(new_phone)

        # Predict price
        price = model.predict(new_phone_scaled)
        st.success(f"The predicted phone price is: RWF {price[0]:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")
