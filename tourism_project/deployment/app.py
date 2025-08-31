import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="hkbindhu/Tourism-Package-Prediction", filename="best_tourism_prediction_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction App")
st.write("The Tourism Package Prediction App predicts whether a customer is likely to purchase a tourism package based on their profile.")
st.write("Please enter the customer details below to check if they are likely to purchase a tourism package.")

# Collect user input
Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100, value=30)
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
CityTier = st.selectbox("City Tier (Tier 1, Tier 2, Tier 3)", ["Tier 1", "Tier 2", "Tier 3"])
Occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Self-Employed", "Other"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of People Visiting (excluding customer)", min_value=0, max_value=10, value=1)
PreferredPropertyStar = st.number_input("Preferred Property Star Rating (1 to 5)", min_value=1, max_value=5, value=3)
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
NumberOfTrips = st.number_input("Number of Trips Taken Annually", min_value=1, max_value=20, value=2)
Passport = st.selectbox("Has Passport?", ["Yes", "No"])
OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting (below 5 years)", min_value=0, max_value=5, value=0)
Designation = st.selectbox("Designation", ["Manager", "Employee", "Self-Employed", "Other"])
MonthlyIncome = st.number_input("Monthly Income (in USD)", min_value=1000, max_value=50000, value=5000)

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "purchase" if prediction == 1 else "not purchase"
    st.write(f"Based on the information provided, the customer is likely to {result} a tourism package.")
