# Heart Disease Prediction App

This repository contains a complete pipeline for predicting heart disease risk using BRFSS 2015 data:
1. Exploratory Data Analysis & Model Training (in `notebooks/EDA_DataPrep_Train.ipynb`).
2. A saved XGBoost model (`models/xgb_model_v1.0.pkl`).
3. A Streamlit app (`app/app.py`) that loads the model and provides a web interface.

---

## Folder Structure

heart-disease-prediction/

├── app/

│ ├── app.py

│ ├── requirements.txt

│ └── README.md

├── data/

│ └── heart_disease_health_indicators_BRFSS2015.csv

├── models/

│ └── xgb_model_v1.0.pkl

├── notebooks/

│ └── EDA_DataPrep_Train.ipynb

└── README.md

---

## How to Get Started

1. **Clone this repository**  
   ```bash
   git clone https://github.com/<YOUR_USERNAME>/heart-disease-prediction.git
   cd heart-disease-prediction
   
2. **Install dependencies for the Streamlit app**
    ```bash
    cd app
    pip install -r requirements.txt

4. **Run the Streamlit App**
   streamlit run app.py

   - A browser window should open at http://localhost:8501.
   - Input patient information on the sidebar and view the predicted risk.
