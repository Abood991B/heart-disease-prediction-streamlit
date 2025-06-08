import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# --- MODEL LOADING ---
# Load the saved model. Make sure the path is correct.
try:
    model = joblib.load('models/gb_model_v1.0.pkl')
except FileNotFoundError:
    st.error("Model file not found. Make sure 'models/gb_model_v1.0.pkl' is in the correct directory.")
    st.stop()


# --- SIDEBAR ---
with st.sidebar:
    st.title("❤️ About this Project")
    st.info(
        "This application uses a Gradient Boosting machine learning model to predict "
        "the likelihood of heart disease. The model was trained on the BRFSS 2015 dataset. "
        "This is a portfolio project to demonstrate an end-to-end data science workflow."
    )
    st.write("---")
    st.write("**Connect with me:**")
    st.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-Profile-blue?style=for-the-badge&logo=github)](https://github.com/Abood991B)"
    )


# --- MAIN PAGE ---
st.title("Heart Disease Early Prediction App 🩺")
st.subheader("Enter patient information to predict the likelihood of a heart disease or attack.")

# Use columns for a better layout
col1, col2 = st.columns(2)

with col1:
    st.header("👤 Personal & Health Info")
    age = st.number_input('Age', min_value=18, max_value=86, value=30)
    sex = st.radio('Sex', ['Male', 'Female'], horizontal=True)
    bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=25.0)
    high_bp = st.selectbox('High Blood Pressure', ['Yes', 'No'])
    high_chol = st.selectbox('High Cholesterol', ['Yes', 'No'])
    chol_check = st.selectbox('Cholesterol Checked', ['Yes', 'No'])
    stroke = st.selectbox('Stroke', ['Yes', 'No'])
    diabetes = st.selectbox('Diabetes', ['No Diabetes', 'Type 1 Diabetes', 'Type 2 Diabetes'])
    diff_walk = st.selectbox('Difficulty in Walking', ['Yes', 'No'])

with col2:
    st.header("🏃 Lifestyle & Behavior")
    smoker = st.selectbox('Smoker', ['Yes', 'No'])
    phy_activity = st.selectbox('Physical activity in past 30 days?', ['Yes', 'No'])
    consume_fruits = st.selectbox('Consume Fruits 1 or more times per day', ['Yes', 'No'])
    consume_veggies = st.selectbox('Consume Veggies 1 or more times per day', ['Yes', 'No'])
    hvy_alcohol_consump = st.selectbox('Heavy Alcohol Consumption', ['Yes', 'No'])
    any_healthcare = st.selectbox('Any Healthcare', ['Yes', 'No'])
    no_docbc_cost = st.selectbox('Unable to afford doctor check-ups?', ['Yes', 'No'])
    gen_hlth = st.selectbox('General Health', ['Excellent', 'Very good', 'Good', 'Fair', 'Poor'])

st.write("---")

st.header("📊 Additional Information")
col3, col4, col5 = st.columns(3)

with col3:
    ment_hlth = st.slider('Days mental health not good (last 30 days)', min_value=0, max_value=30, value=15)
with col4:
    phys_hlth = st.slider('Days physical health not good (last 30 days)', min_value=0, max_value=30, value=15)
with col5:
    education = st.selectbox('Education', ['Kinder', 'Primary', 'Junior', 'High/Diploma', 'Bachelor', 'Postgrad'])
    income = st.selectbox('Income', ['Less than $10,000', '$10,000-$24,999', '$25,000-$34,999', '$35,000-$49,999', '$50,000-$74,999', '$75,000 or more'])

# --- PREDICTION LOGIC ---
if st.button('**Get Prediction**', use_container_width=True):
    # Map categorical inputs to numerical values using your original mappings
    gen_hlth_mapping = {'Excellent': 1, 'Very good': 2, 'Good': 3, 'Fair': 4, 'Poor': 5}
    sex_mapping = {'Male': 1, 'Female': 0}
    binary_mapping = {'Yes': 1, 'No': 0}
    diabetes_mapping = {'No Diabetes': 0, 'Type 1 Diabetes': 1, 'Type 2 Diabetes': 2}
    education_mapping = {'Kinder': 1, 'Primary': 2, 'Junior': 3, 'High/Diploma': 4, 'Bachelor': 5, 'Postgrad': 6}
    income_mapping = {'Less than $10,000': 1, '$10,000-$24,999': 2, '$25,000-$34,999': 3, '$35,000-$49,999': 4, '$50,000-$74,999': 5, '$75,000 or more': 6}

    # Preprocess the input data (Your original logic)
    phy_activity_val = binary_mapping[phy_activity]
    gen_hlth_val = gen_hlth_mapping[gen_hlth]
    sex_val = sex_mapping[sex]
    high_bp_val = binary_mapping[high_bp]
    high_chol_val = binary_mapping[high_chol]
    chol_check_val = binary_mapping[chol_check]
    smoker_val = binary_mapping[smoker]
    stroke_val = binary_mapping[stroke]
    diabetes_val = diabetes_mapping[diabetes]
    consume_fruits_val = binary_mapping[consume_fruits]
    consume_veggies_val = binary_mapping[consume_veggies]
    hvy_alcohol_consump_val = binary_mapping[hvy_alcohol_consump]
    any_healthcare_val = binary_mapping[any_healthcare]
    no_docbc_cost_val = binary_mapping[no_docbc_cost]
    diff_walk_val = binary_mapping[diff_walk]
    education_val = education_mapping[education]
    income_val = income_mapping[income]

    # Automatically categorize age (Your original logic)
    age_group = int((age - 18) / 5) + 1

    # Create the feature list in the exact order the model expects
    features = [
        high_bp_val, high_chol_val, chol_check_val, bmi, smoker_val,
        stroke_val, diabetes_val, phy_activity_val, consume_fruits_val,
        consume_veggies_val, hvy_alcohol_consump_val, any_healthcare_val,
        no_docbc_cost_val, gen_hlth_val, ment_hlth, phys_hlth,
        diff_walk_val, sex_val, age_group, education_val, income_val
    ]

    # Make predictions using the loaded model
    prediction_proba = model.predict_proba([features])
    prediction = model.predict([features])

    # Display the result
    st.write("---")
    st.header("📈 Prediction Result")

    likelihood_percentage = prediction_proba[0][1] * 100

    if prediction[0] == 1:
        st.error(f"High Likelihood of Heart Disease ({likelihood_percentage:.2f}%)")
        st.warning(
            "This result indicates a significant risk. It is strongly recommended to consult with a "
            "healthcare professional for a proper evaluation."
        )
    else:
        st.success(f"Low Likelihood of Heart Disease ({likelihood_percentage:.2f}%)")
        st.info(
            "This is a positive sign! Continue to maintain a healthy lifestyle and have regular check-ups."
        )

    st.metric(label="Probability of Heart Disease", value=f"{likelihood_percentage:.2f}%")


# --- DISCLAIMER ---
st.write("---")
st.caption(
    "**Disclaimer:** This prediction is based on a machine learning model and is not a substitute for "
    "professional medical advice. The model has limitations and its accuracy is not 100%. "
    "Please consult a doctor for any health concerns."
)
