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
# 2. USER INTERFACE (UI) WITH TABS
# ==========================================
st.title("📊 Employee Attrition Risk Assessment Dashboard")
st.markdown("""
This production tool uses an optimized Logistic Regression engine to assess employee flight risk based on comprehensive demographic, financial, job role, and workplace satisfaction indicators.
""")
st.write("---")

# Create multi-tab layout for cleaner organization
tab1, tab2, tab3 = st.tabs(["👤 Personal & Demographics", "💼 Job & Workplace", "⭐ Satisfaction & Survey"])

with tab1:
    st.subheader("Employee Demographics & Career Tenure")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", min_value=18, max_value=65, value=35)
        gender = st.selectbox("Gender", options=["Male", "Female"])
        marital_status = st.selectbox("Marital Status", options=["Single", "Married", "Divorced"])
        distance_from_home = st.slider("Distance From Home (miles)", min_value=1, max_value=30, value=5)
        
    with col2:
        total_working_years = st.slider("Total Working Years", min_value=0, max_value=40, value=10)
        years_at_company = st.slider("Years at Company", min_value=0, max_value=40, value=5)
        years_in_current_role = st.slider("Years in Current Role", min_value=0, max_value=20, value=3)
        years_with_curr_manager = st.slider("Years with Current Manager", min_value=0, max_value=20, value=3)

with tab2:
    st.subheader("Job Role, Department & Compensation")
    col3, col4 = st.columns(2)
    
    with col3:
        department = st.selectbox("Department", options=["Research & Development", "Sales", "Human Resources"])
        job_role = st.selectbox("Job Role", options=[
            "Sales Executive", "Research Scientist", "Laboratory Technician", 
            "Manufacturing Director", "Healthcare Representative", "Manager", 
            "Sales Representative", "Research Director", "Human Resources"
        ])
        job_level = st.slider("Job Level", min_value=1, max_value=5, value=2)
        monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=25000, value=5000, step=500)
        
    with col4:
        overtime = st.selectbox("Does the employee work Overtime?", options=["No", "Yes"])
        business_travel = st.selectbox("Frequency of Business Travel", options=["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
        stock_option_level = st.selectbox("Stock Option Level", options=[0, 1, 2, 3], index=1)
        num_companies_worked = st.slider("Number of Companies Worked Before", min_value=0, max_value=9, value=2)

with tab3:
    st.subheader("Satisfaction Surveys & Performance Metrics")
    col5, col6 = st.columns(2)
    
    with col5:
        job_satisfaction = st.selectbox("Job Satisfaction (1 = Low, 4 = Very High)", options=[1, 2, 3, 4], index=2)
        environment_satisfaction = st.selectbox("Environment Satisfaction (1 = Low, 4 = Very High)", options=[1, 2, 3, 4], index=2)
        relationship_satisfaction = st.selectbox("Relationship Satisfaction (1 = Low, 4 = Very High)", options=[1, 2, 3, 4], index=2)
        work_life_balance = st.selectbox("Work-Life Balance (1 = Bad, 4 = Best)", options=[1, 2, 3, 4], index=2)
        
    with col6:
        education_field = st.selectbox("Education Field", options=["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"])
        education_level = st.selectbox("Education Level (1 to 5)", options=[1, 2, 3, 4, 5], index=2)
        performance_rating = st.selectbox("Performance Rating (3 = Excellent, 4 = Outstanding)", options=[3, 4], index=0)
        training_times_last_year = st.slider("Training Times Last Year", min_value=0, max_value=6, value=2)
        years_since_last_promotion = st.slider("Years Since Last Promotion", min_value=0, max_value=15, value=1)

st.write("---")

# ==========================================
# 3. DATA PROCESSING & PREDICTION PIPELINE
# ==========================================
if st.button("📊 Run Risk Assessment", type="primary", use_container_width=True):
    
    # 1. Map user selections to model features & encoding structure
    production_row = {
        'Age': age,
        'BusinessTravel': 0 if business_travel == "Non-Travel" else (1 if business_travel == "Travel_Rarely" else 2),
        'DailyRate': 800,  # Baseline median default for unexposed micro-features
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
        'NumCompaniesWorked': num_companies_worked,
        'OverTime': 1 if overtime == "Yes" else 0,
        'PercentSalaryHike': 15,
        'PerformanceRating': performance_rating,
        'RelationshipSatisfaction': relationship_satisfaction,
        'StockOptionLevel': stock_option_level,
        'TotalWorkingYears': total_working_years,
        'TrainingTimesLastYear': training_times_last_year,
        'WorkLifeBalance': work_life_balance,
        'YearsAtCompany': years_at_company,
        'YearsInCurrentRole': years_in_current_role,
        'YearsSinceLastPromotion': years_since_last_promotion,
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
    
    # ==========================================
    # 4. PRODUCTION-GRADE DISPLAY & METRICS
    # ==========================================
    st.write("### 🛑 Risk Assessment Metrics")
    
    risk_percent = risk_score * 100
    baseline_attrition = 16.0
    variance = risk_percent - baseline_attrition
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric(
            label="Computed Flight Risk", 
            value=f"{risk_percent:.1f}%", 
            delta=f"{variance:+.1f}% vs Company Avg",
            delta_color="inverse"
        )
    with col_m2:
        status_string = "⚠️ HIGH RISK" if risk_score >= 0.50 else "✅ STABLE"
        st.metric(label="Retention Status", value=status_string)
    
    if risk_score >= 0.50:
        st.progress(risk_score, text=f"Critical Alert Zone ({risk_percent:.1f}%)")
    else:
        st.progress(risk_score, text=f"Safe Operational Zone ({risk_percent:.1f}%)")
        
    st.write("---")
    st.write("### 📊 Key Risk Drivers Analysis")
    st.markdown("This chart isolates how the employee's active profile configuration mathematically impacts their retention probability.")

    # Dynamic Visualization Data Extraction
    coefficients = model.coef_[0]
    feature_names = list(model.feature_names_in_)
    weight_map = dict(zip(feature_names, coefficients))

    ot_yes_key = next((k for k in weight_map.keys() if 'overtime' in k.lower() and 'yes' in k.lower()), None)
    overtime_impact = weight_map[ot_yes_key] if (overtime == "Yes" and ot_yes_key) else 0.0

    dist_key = next((k for k in weight_map.keys() if 'distance' in k.lower()), None)
    tenure_key = next((k for k in weight_map.keys() if 'yearsatcompany' in k.lower()), None)
    role_key = next((k for k in weight_map.keys() if 'currentrole' in k.lower()), None)
    manager_key = next((k for k in weight_map.keys() if 'manager' in k.lower()), None)

    distance_impact = weight_map[dist_key] * distance_from_home if dist_key else 0.0
    tenure_impact = weight_map[tenure_key] * years_at_company if tenure_key else 0.0
    role_time_impact = weight_map[role_key] * years_in_current_role if role_key else 0.0
    manager_impact = weight_map[manager_key] * years_with_curr_manager if manager_key else 0.0

    impact_data = {
        "Factor": [
            "Overtime Engagement", 
            "Commute Stress (Distance)", 
            "Tenure at Company", 
            "Time in Current Role", 
            "Relationship with Manager"
        ],
        "Impact Score": [
            overtime_impact,
            distance_impact,
            tenure_impact,
            role_time_impact,
            manager_impact
        ]
    }
    
    chart_df = pd.DataFrame(impact_data).set_index("Factor")
    st.bar_chart(chart_df, y="Impact Score", use_container_width=True)
    
    st.caption("💡 **How to read this chart:** Positive values (bars pointing right) increase flight risk. Negative values (bars pointing left) act as protective retention anchors.")
    st.write("---")
    
    st.info(f"""
    **Operational Deep-Dive:**
    * This employee profile yields a computed probability of **{risk_score:.1%}** against an organizational baseline of **{baseline_attrition}%**.
    * {'Critical Factor: Overtime engagement increases baseline exit probabilities drastically.' if overtime == 'Yes' else 'Protective Factor: Lack of overtime drastically lowers burnout risk indicators.'}
    """)