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
# Load the saved Gradient Boosting model
# Ensure the path is correct based on your project structure
try:
    model = joblib.load('models/gb_model_v1.0.pkl')
except FileNotFoundError:
    st.error("Model file not found. Make sure 'models/gb_model_v1.0.pkl' is in the correct directory.")
    st.stop()


# --- HELPER FUNCTIONS ---
# Helper function to preprocess user inputs
def preprocess_input(data):
    """Maps categorical user inputs to numerical values for the model."""
    # Mapping dictionaries
    gen_hlth_mapping = {'Excellent': 1, 'Very good': 2, 'Good': 3, 'Fair': 4, 'Poor': 5}
    sex_mapping = {'Male': 1, 'Female': 0}
    binary_mapping = {'Yes': 1, 'No': 0}
    diabetes_mapping = {'No Diabetes': 0, 'Type 1 Diabetes': 1, 'Type 2 Diabetes': 2}
    education_mapping = {'Kindergarten or less': 1, 'Primary School': 2, 'Middle School': 3, 'High School/GED': 4, 'College': 5, 'Postgraduate': 6}
    income_mapping = {
        'Less than $10,000': 1, '$10,000 to $14,999': 2, '$15,000 to $19,999': 3,
        '$20,000 to $24,999': 4, '$25,000 to $34,999': 5, '$35,000 to $49,999': 6,
        '$50,000 to $74,999': 7, '$75,000 or more': 8
    }

    # Automatically categorize age (assuming the model was trained on these categories)
    # Age categories: 1 = 18-24, 2 = 25-29, ..., 13 = 80+
    if 18 <= data['age'] <= 24: age_group = 1
    elif 25 <= data['age'] <= 29: age_group = 2
    elif 30 <= data['age'] <= 34: age_group = 3
    elif 35 <= data['age'] <= 39: age_group = 4
    elif 40 <= data['age'] <= 44: age_group = 5
    elif 45 <= data['age'] <= 49: age_group = 6
    elif 50 <= data['age'] <= 54: age_group = 7
    elif 55 <= data['age'] <= 59: age_group = 8
    elif 60 <= data['age'] <= 64: age_group = 9
    elif 65 <= data['age'] <= 69: age_group = 10
    elif 70 <= data['age'] <= 74: age_group = 11
    elif 75 <= data['age'] <= 79: age_group = 12
    else: age_group = 13
    
    # Create the feature list in the exact order the model expects
    features = [
        binary_mapping[data['high_bp']],
        binary_mapping[data['high_chol']],
        binary_mapping[data['chol_check']],
        data['bmi'],
        binary_mapping[data['smoker']],
        binary_mapping[data['stroke']],
        diabetes_mapping[data['diabetes']],
        binary_mapping[data['phy_activity']],
        binary_mapping[data['consume_fruits']],
        binary_mapping[data['consume_veggies']],
        binary_mapping[data['hvy_alcohol_consump']],
        binary_mapping[data['any_healthcare']],
        binary_mapping[data['no_docbc_cost']],
        gen_hlth_mapping[data['gen_hlth']],
        data['ment_hlth'],
        data['phys_hlth'],
        binary_mapping[data['diff_walk']],
        sex_mapping[data['sex']],
        age_group,
        education_mapping[data['education']],
        income_mapping[data['income']]
    ]
    
    return np.array(features).reshape(1, -1)

# --- SIDEBAR ---
with st.sidebar:
    st.title("❤️ About this Project")
    st.info(
        "This application uses a Gradient Boosting machine learning model to predict "
        "the likelihood of heart disease. The model was trained on the BRFSS 2015 dataset. "
        "This is a project to demonstrate an end-to-end data science workflow."
    )
    st.write("---")
    st.write("**Connect with me:**")
    st.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-Profile-blue?style=for-the-badge&logo=github)](https://github.com/Abood991B)"
    )
    st.markdown(
        "[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/abdulrahman-baidaq-05781b2a9)"
    )

# --- MAIN PAGE ---
st.title("Heart Disease Early Prediction App 🩺")
st.subheader("Enter patient information to predict the likelihood of a heart disease or attack.")

# Use columns for a better layout
col1, col2, col3 = st.columns(3)

with col1:
    st.header("👤 Demographics")
    age = st.number_input('Age', min_value=18, max_value=100, value=45)
    sex = st.radio('Sex', ['Female', 'Male'], horizontal=True)
    bmi = st.number_input('Body Mass Index (BMI)', min_value=10.0, max_value=100.0, value=25.0, step=0.1)
    
with col2:
    st.header("🩺 Health Indicators")
    high_bp = st.radio('High Blood Pressure?', ('Yes', 'No'), horizontal=True)
    high_chol = st.radio('High Cholesterol?', ('Yes', 'No'), horizontal=True)
    chol_check = st.radio('Cholesterol Checked in last 5 years?', ('Yes', 'No'), horizontal=True)
    stroke = st.radio('History of Stroke?', ('Yes', 'No'), horizontal=True)
    diabetes = st.selectbox('Diabetes Status', ('No Diabetes', 'Type 1 Diabetes', 'Type 2 Diabetes'))

with col3:
    st.header("🏃 Lifestyle & Behavior")
    smoker = st.radio('Are you a smoker?', ('Yes', 'No'), horizontal=True)
    phy_activity = st.radio('Recent Physical Activity?', ('Yes', 'No'), horizontal=True)
    hvy_alcohol_consump = st.radio('Heavy Alcohol Consumption?', ('Yes', 'No'), horizontal=True, help="Men: 14+ drinks/week, Women: 7+ drinks/week")
    consume_fruits = st.radio('Consume Fruits 1+ times/day?', ('Yes', 'No'), horizontal=True)
    consume_veggies = st.radio('Consume Veggies 1+ times/day?', ('Yes', 'No'), horizontal=True)

st.write("---")

st.header("💬 Subjective & Healthcare Access")
col4, col5 = st.columns(2)

with col4:
    gen_hlth = st.select_slider(
        'General Health Perception',
        options=['Excellent', 'Very good', 'Good', 'Fair', 'Poor'],
        help="How would you rate your general health? 1=Excellent, 5=Poor"
    )
    ment_hlth = st.slider('Days of Poor Mental Health (last 30 days)', min_value=0, max_value=30, value=5)
    phys_hlth = st.slider('Days of Poor Physical Health (last 30 days)', min_value=0, max_value=30, value=5)

with col5:
    any_healthcare = st.radio('Have any kind of health care coverage?', ('Yes', 'No'), horizontal=True)
    no_docbc_cost = st.radio('Avoided doctor due to cost in past year?', ('Yes', 'No'), horizontal=True)
    diff_walk = st.radio('Have serious difficulty walking or climbing stairs?', ('Yes', 'No'), horizontal=True)

st.write("---")

st.header("🎓 Socioeconomic Status")
col6, col7 = st.columns(2)

with col6:
    education = st.selectbox('Highest Education Level', ('Kindergarten or less', 'Primary School', 'Middle School', 'High School/GED', 'College', 'Postgraduate'))
    
with col7:
    income = st.selectbox('Annual Household Income', (
        'Less than $10,000', '$10,000 to $14,999', '$15,000 to $19,999',
        '$20,000 to $24,999', '$25,000 to $34,999', '$35,000 to $49,999',
        '$50,000 to $74,999', '$75,000 or more'
    ))

# --- PREDICTION ---
if st.button('**Get Prediction**', use_container_width=True):
    # Prepare the input data
    user_data = {
        'age': age, 'sex': sex, 'bmi': bmi, 'high_bp': high_bp, 'high_chol': high_chol,
        'chol_check': chol_check, 'smoker': smoker, 'stroke': stroke, 'diabetes': diabetes,
        'phy_activity': phy_activity, 'consume_fruits': consume_fruits, 'consume_veggies': consume_veggies,
        'hvy_alcohol_consump': hvy_alcohol_consump, 'any_healthcare': any_healthcare,
        'no_docbc_cost': no_docbc_cost, 'diff_walk': diff_walk, 'gen_hlth': gen_hlth,
        'ment_hlth': ment_hlth, 'phys_hlth': phys_hlth, 'education': education, 'income': income
    }
    
    # Preprocess and predict
    processed_features = preprocess_input(user_data)
    prediction_proba = model.predict_proba(processed_features)
    prediction = model.predict(processed_features)
    
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
        
    # Optional: Show detailed probability
    st.metric(label="Probability of Heart Disease", value=f"{likelihood_percentage:.2f}%")


# --- DISCLAIMER ---
st.write("---")
st.caption(
    "**Disclaimer:** This prediction is based on a machine learning model and is not a substitute for "
    "professional medical advice. The model has limitations and its accuracy is not 100%. "
    "Please consult a doctor for any health concerns."
)
