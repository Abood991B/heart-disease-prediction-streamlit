import streamlit as st
import joblib
import pandas as pd

# Load the saved model
model = joblib.load('models/xgb_model_v1.0.pkl')

# Define the Streamlit app
def main():
    st.title('Heart Disease Early Prediction App')
    st.write('Enter patient information to predict the likelihood of a heart disease or attack:')
    
    # Collect user input
    age = st.number_input('Age', min_value=18, max_value=86, value=30)
    sex = st.radio('Sex', ['Male', 'Female'])
    bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=25.0)

    high_bp = st.selectbox('High Blood Pressure', ['Yes', 'No'])
    high_chol = st.selectbox('High Cholesterol', ['Yes', 'No'])
    chol_check = st.selectbox('Cholesterol Checked', ['Yes', 'No'])
    smoker = st.selectbox('Smoker', ['Yes', 'No'])
    stroke = st.selectbox('Stroke', ['Yes', 'No'])
    diabetes = st.selectbox('Diabetes', ['No Diabetes', 'Type 1 Diabetes', 'Type 2 Diabetes'])
    
    phy_activity = st.selectbox('Have you done physical activity or exercise during the past 30 days other than your regular job?', ['Yes', 'No'])
    consume_fruits = st.selectbox('Consume Fruits 1 or more times per day', ['Yes', 'No'])
    consume_veggies = st.selectbox('Consume Veggies 1 or more times per day', ['Yes', 'No'])
    hvy_alcohol_consump = st.selectbox('Heavy Alcohol Consumption', ['Yes', 'No'])
    any_healthcare = st.selectbox('Any Healthcare', ['Yes', 'No'])
    no_docbc_cost = st.selectbox('Unable to afford regular doctor check-ups ', ['Yes', 'No'])
    diff_walk = st.selectbox('Difficulty in Walking', ['Yes', 'No'])
    gen_hlth = st.selectbox('General Health', ['Excellent', 'Very good', 'Good', 'Fair', 'Poor'])
    ment_hlth = st.slider('How many days during the past 30 days was your mental health not good?', min_value=0, max_value=30, value=15)
    phys_hlth = st.slider('How many days during the past 30 days was your physical health not good?', min_value=0, max_value=30, value=15)


    education = st.selectbox('Education', ['Kinder', 'Primary', 'Junior', 'High/Diploma', 'Bachelor', 'Postgrad'])
    income = st.selectbox('Income', ['Less than $10,000', '$10,000-$24,999', '$25,000-$34,999', '$35,000-$49,999', '$50,000-$74,999', '$75,000 or more'])
    
    # Map categorical inputs to numerical values
    gen_hlth_mapping = {'Excellent': 1, 'Very good': 2, 'Good': 3, 'Fair': 4, 'Poor': 5}
    sex_mapping = {'Male': 1, 'Female': 0}
    binary_mapping = {'Yes': 1, 'No': 0}
    diabetes_mapping = {'No Diabetes': 0, 'Type 1 Diabetes': 1, 'Type 2 Diabetes': 2}
    education_mapping = {'Kinder': 1, 'Primary': 2, 'Junior': 3, 'High/Diploma': 4, 'Bachelor': 5, 'Postgrad': 6}
    income_mapping = {'Less than $10,000': 1, '$10,000-$24,999': 2, '$25,000-$34,999': 3, '$35,000-$49,999': 4, '$50,000-$74,999': 5, '$75,000 or more': 6}
    
    # Preprocess the input data
    phy_activity = binary_mapping[phy_activity]
    gen_hlth = gen_hlth_mapping[gen_hlth]
    sex = sex_mapping[sex]
    high_bp = binary_mapping[high_bp]
    high_chol = binary_mapping[high_chol]
    chol_check = binary_mapping[chol_check]
    smoker = binary_mapping[smoker]
    stroke = binary_mapping[stroke]
    diabetes = diabetes_mapping[diabetes]
    consume_fruits = binary_mapping[consume_fruits]
    consume_veggies = binary_mapping[consume_veggies]
    hvy_alcohol_consump = binary_mapping[hvy_alcohol_consump]
    any_healthcare = binary_mapping[any_healthcare]
    no_docbc_cost = binary_mapping[no_docbc_cost]
    diff_walk = binary_mapping[diff_walk]
    education = education_mapping[education]
    income = income_mapping[income]

    
    # Automatically categorize age into one of the 13 groups
    age_group = int((age - 18) / 5) + 1
    
    # Make predictions using the loaded model
    prediction = model.predict([[high_bp, high_chol, chol_check, bmi, smoker, stroke, diabetes, phy_activity, consume_fruits, consume_veggies, hvy_alcohol_consump, any_healthcare, no_docbc_cost, gen_hlth, ment_hlth, phys_hlth, diff_walk, sex, age_group, education, income]])
    
    # Display the prediction result
    if prediction[0] == 1:
        st.error('High likelihood of heart disease or attack!')
    else:
        st.success('Low likelihood of heart disease or attack.')

if __name__ == '__main__':
    main()
