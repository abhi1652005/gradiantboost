import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Load Model
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    # Ensure your model file name matches (e.g., 'grndiantr.pkl')
    with open('grndiantr.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# App Layout
# -----------------------------------------------------------------------------
st.title("💼 Employee Salary Prediction Model")
st.write("Enter the employee details below to predict the target metric.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Employee Details")
    age = st.number_input("Age", min_value=18, max_value=70, value=30, step=1)
    experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
    
    # Define options based on your training categories
    gender_options = ["Male", "Female", "Other"]
    gender = st.selectbox("Gender", options=gender_options)
    
    education_options = ["High School", "Bachelor's", "Master's", "PhD"]
    education = st.selectbox("Education Level", options=education_options)

with col2:
    st.subheader("Location & Work")
    department_options = ["Engineering", "HR", "Sales", "Marketing", "Finance", "Operations"]
    department = st.selectbox("Department", options=department_options)
    
    city_options = ["New York", "San Francisco", "Chicago", "Austin", "Seattle", "Remote"]
    city = st.selectbox("City", options=city_options)

# -----------------------------------------------------------------------------
# Encoding Mappings
# -----------------------------------------------------------------------------
# NOTE: If you used a specific LabelEncoder mapping during training, 
# match those integer values here. Below is a standard mapping example:
department_mapping = {val: idx for idx, val in enumerate(department_options)}
education_mapping = {val: idx for idx, val in enumerate(education_options)}
gender_mapping = {val: idx for idx, val in enumerate(gender_options)}
city_mapping = {val: idx for idx, val in enumerate(city_options)}

# -----------------------------------------------------------------------------
# Prediction Logic
# -----------------------------------------------------------------------------
st.markdown("---")

if st.button("Predict Target Value"):
    try:
        # Convert user selections to the numerical format expected by the model
        input_data = pd.DataFrame([{
            'Department': department_mapping[department],
            'Experience_Years': experience,
            'Education_Level': education_mapping[education],
            'Age': age,
            'Gender': gender_mapping[gender],
            'City': city_mapping[city]
        }])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display Result
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader("Predicted Result")
        st.markdown(f"# **{prediction:,.2f}**")
        st.markdown("</div>", unsafe_allow_html=True)
        
    except Exception as err:
        st.error(f"Prediction failed. Error: {err}")