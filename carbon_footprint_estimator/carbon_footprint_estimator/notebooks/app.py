import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("carbon_emission_model.pkl")

# Load feature columns (from training)
X_columns = joblib.load("X_columns.pkl")
 # Saved during training

# ---- Prediction Function ----
def predict_emission(soil_pH, soil_moisture, temperature, rainfall, crop_type, fertilizer, pesticide, crop_yield):
    # Sanitize crop type input
    crop_type = str(crop_type).strip().title()  # e.g. "rice " -> "Rice"
    
    # Prepare input
    
    input_data = pd.DataFrame({
        "Soil_pH": [soil_pH],
        "Soil_Moisture": [soil_moisture],
        "Temperature_C": [temperature],
        "Rainfall_mm": [rainfall],
        "Crop_type": [crop_type],
        "Fertilizer_Usage_kg": [fertilizer],
        "Pesticide_Usage_kg": [pesticide],
        "Crop_Yield_ton": [crop_yield]
    })

    # One-hot encode Crop_Type
    input_encoded = pd.get_dummies(input_data, columns=["Crop_Type"], drop_first=True)

    # Align with training columns
    for col in X_columns:
        if col not in input_encoded:
            input_encoded[col] = 0
    input_encoded = input_encoded[X_columns]

    # Predict emission
    emission_pred = model.predict(input_encoded)[0]

    # Suggest measures
    if emission_pred < 50:
        measure = "✅ Sustainable – Maintain practices"
    elif 50 <= emission_pred < 100:
        measure = "⚠️ Moderate – Optimize fertilizer & irrigation"
    else:
        measure = "❌ High – Reduce chemical usage, adopt organic methods"

    return emission_pred, measure


# ---- Streamlit UI ----
st.title("🌱 Smart Carbon Emission Advisor for Farmers")

st.write("Provide your farming details below to estimate carbon emissions and get recommendations.")

soil_pH = st.number_input("Soil pH", 4.0, 9.0, 6.5)
soil_moisture = st.number_input("Soil Moisture (%)", 5, 100, 25)
temperature = st.number_input("Temperature (°C)", 5, 50, 30)
rainfall = st.number_input("Rainfall (mm)", 0, 500, 120)
crop_type = st.selectbox("Crop Type", ["Wheat", "Rice", "Maize", "Pulses"])
fertilizer = st.number_input("Fertilizer Usage (kg)", 0, 200, 60)
pesticide = st.number_input("Pesticide Usage (kg)", 0, 100, 15)
crop_yield = st.number_input("Crop Yield (ton)", 0, 50, 5)

if st.button("Predict Carbon Emission"):
    emission, recommendation = predict_emission(
        soil_pH, soil_moisture, temperature, rainfall,
        crop_type, fertilizer, pesticide, crop_yield
    )
    
    st.success(f"🌍 Predicted Carbon Emission: **{emission:.2f} units**")
    st.info(f"💡 Recommendation: {recommendation}")
