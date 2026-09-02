import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. PAGE CONFIGURATION & ASSET LOADING
# ==========================================
st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="📊",
    layout="centered"
)

@st.cache_resource
def load_production_artifacts():
    model = joblib.load('attrition_model.pkl')
    scaler = joblib.load('numerical_scaler.pkl')
    return model, scaler

try:
    model, scaler = load_production_artifacts()
except Exception as e:
    st.error("Error loading model artifacts. Ensure 'attrition_model.pkl' and 'numerical_scaler.pkl' are in the same directory.")
    st.stop()

# ==========================================
# 2. USER INTERFACE (UI) WITH EXACT TABS
# ==========================================
st.title("📊 Employee Attrition Risk Assessment")
st.markdown("""
This production tool uses an optimized Logistic Regression engine to assess employee flight risk based on key demographic, financial, and workplace indicators.
""")
st.write("---")

# Exact tab naming matching your UI screenshot
tab1, tab2, tab3 = st.tabs([
    "📊 Personal & History", 
    "💼 Job Role Details", 
    "❤️ Satisfaction Survey"
])

with tab1:
    st.subheader("Employee Demographics & History")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", min_value=18, max_value=65, value=30)
        distance_from_home = st.slider("Distance From Home (KM)", min_value=1, max_value=30, value=5)
        
    with col2:
        gender = st.selectbox("Gender", options=["Male", "Female"])
        marital_status = st.selectbox("Marital Status", options=["Single", "Married", "Divorced"])

    st.write("---")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        total_working_years = st.slider("Total Working Years", min_value=0, max_value=40, value=10)
        years_at_company = st.slider("Years at Company", min_value=0, max_value=40, value=5)
    with col_t2:
        education_field = st.selectbox("Education Field", options=["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"])
        education_level = st.selectbox("Education Level (1 to 5)", options=[1, 2, 3, 4, 5], index=2)

with tab2:
    st.subheader("Job Role & Workplace Conditions")
    col3, col4 = st.columns(2)
    
    with col3:
        department = st.selectbox("Department", options=["Research & Development", "Sales", "Human Resources"])
        job_role = st.selectbox("Job Role", options=[
            "Research Scientist", "Sales Executive", "Laboratory Technician", 
            "Manufacturing Director", "Healthcare Representative", "Manager", 
            "Sales Representative", "Research Director", "Human Resources"
        ])
        monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=25000, value=5000, step=500)
        
    with col4:
        overtime = st.selectbox("Does the employee work Overtime?", options=["No", "Yes"])
        business_travel = st.selectbox("Frequency of Business Travel", options=["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
        job_level = st.slider("Job Level", min_value=1, max_value=5, value=2)

    st.write("---")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        years_in_current_role = st.slider("Years in Current Role", min_value=0, max_value=20, value=3)
    with col_r2:
        years_with_curr_manager = st.slider("Years with Current Manager", min_value=0, max_value=20, value=3)

with tab3:
    st.subheader("Satisfaction & Performance Survey")
    col5, col6 = st.columns(2)
    
    with col5:
        job_satisfaction = st.selectbox("Job Satisfaction (1 = Low, 4 = Very High)", options=[1, 2, 3, 4], index=2)
        environment_satisfaction = st.selectbox("Environment Satisfaction (1 = Low, 4 = Very High)", options=[1, 2, 3, 4], index=2)
        
    with col6:
        relationship_satisfaction = st.selectbox("Relationship Satisfaction (1 = Low, 4 = Very High)", options=[1, 2, 3, 4], index=2)
        work_life_balance = st.selectbox("Work-Life Balance (1 = Bad, 4 = Best)", options=[1, 2, 3, 4], index=2)

st.write("---")

# ==========================================
# 3. DATA PROCESSING & PREDICTION PIPELINE
# ==========================================
if st.button("📊 Run Risk Assessment", type="primary", use_container_width=True):
    
    production_row = {
        'Age': age,
        'BusinessTravel': 0 if business_travel == "Non-Travel" else (1 if business_travel == "Travel_Rarely" else 2),
        'DailyRate': 800,
        'DistanceFromHome': distance_from_home,
        'Education': education_level,
        'EnvironmentSatisfaction': environment_satisfaction, 
        'Gender': 1 if gender == "Male" else 0, 
        'HourlyRate': 65,
        'JobInvolvement': 3,
        'JobLevel': job_level,        
        'JobSatisfaction': job_satisfaction,
        'MonthlyIncome': monthly_income,
        'MonthlyRate': 14000,
        'NumCompaniesWorked': 2,
        'OverTime': 1 if overtime == "Yes" else 0,
        'PercentSalaryHike': 15,
        'PerformanceRating': 3,
        'RelationshipSatisfaction': relationship_satisfaction,
        'StockOptionLevel': 1,
        'TotalWorkingYears': total_working_years,
        'TrainingTimesLastYear': 2,
        'WorkLifeBalance': work_life_balance,
        'YearsAtCompany': years_at_company,
        'YearsInCurrentRole': years_in_current_role,
        'YearsSinceLastPromotion': 1,
        'YearsWithCurrManager': years_with_curr_manager,
        
        # Department dummies
        'Department_Research & Development': 1 if department == "Research & Development" else 0, 
        'Department_Sales': 1 if department == "Sales" else 0,
        
        # Education Field dummies
        'EducationField_Life Sciences': 1 if education_field == "Life Sciences" else 0,
        'EducationField_Marketing': 1 if education_field == "Marketing" else 0,
        'EducationField_Medical': 1 if education_field == "Medical" else 0,
        'EducationField_Other': 1 if education_field == "Other" else 0,
        'EducationField_Technical Degree': 1 if education_field == "Technical Degree" else 0,
        
        # Job Role dummies
        'JobRole_Human Resources': 1 if job_role == "Human Resources" else 0,
        'JobRole_Laboratory Technician': 1 if job_role == "Laboratory Technician" else 0,
        'JobRole_Manager': 1 if job_role == "Manager" else 0,
        'JobRole_Manufacturing Director': 1 if job_role == "Manufacturing Director" else 0,
        'JobRole_Research Director': 1 if job_role == "Research Director" else 0,
        'JobRole_Research Scientist': 1 if job_role == "Research Scientist" else 0,
        'JobRole_Sales Executive': 1 if job_role == "Sales Executive" else 0,
        'JobRole_Sales Representative': 1 if job_role == "Sales Representative" else 0,
        
        # Marital Status dummies
        'MaritalStatus_Married': 1 if marital_status == "Married" else 0,
        'MaritalStatus_Single': 1 if marital_status == "Single" else 0
    }
    
    input_df = pd.DataFrame([production_row])
    
    numerical_cols = [
        'Age', 'MonthlyIncome', 'DistanceFromHome', 
        'TotalWorkingYears', 'YearsAtCompany', 
        'YearsInCurrentRole', 'YearsWithCurrManager'
    ]
    
    try:
        scaled_df = input_df.copy()
        scaled_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
        model_features = model.feature_names_in_
        final_encoded_df = scaled_df[model_features]
    except Exception as e:
        st.error(f"Pipeline processing error: {e}")
        st.stop()
        
    probabilities = model.predict_proba(final_encoded_df)[0]
    risk_score = probabilities[1]
    risk_percent = risk_score * 100
    
    # ==========================================
    # 4. MATCHING PHOTO 2 OUTPUT FORMATTING
    # ==========================================
    st.write("## Prediction Result")
    
    if risk_score >= 0.50:
        st.error(f"⚠️ High Attrition Risk: This employee is at risk of leaving. (Risk: {risk_percent:.1f}%)")
    else:
        st.success(f"💚 Low Attrition Risk: This employee is likely to stay. (Risk: {risk_percent:.1f}%)")
        
    st.progress(risk_score)
    
    with st.container():
        st.write("### 📝 Employee Risk Profile Summary")
        st.caption("A consolidated view of the parameters evaluated by the Logistic Regression model.")
        
        col_sum1, col_sum2 = st.columns(2)
        
        with col_sum1:
            st.markdown("#### 👤 Personal & Demographics")
            st.markdown(f"""
            * **Age:** {age} years old
            * **Marital Status:** {marital_status}
            * **Distance From Home:** {distance_from_home} KM
            * **Education Field:** {education_field}
            """)
            
        with col_sum2:
            st.markdown("#### 💼 Job & Sentiment Details")
            st.markdown(f"""
            * **Department:** {department}
            * **Job Role:** {job_role}
            * **Monthly Income:** ${monthly_income:,}
            * **Overtime Required:** {overtime}
            """)