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
# 2. USER INTERFACE (UI) DESIGN
# ==========================================
st.title("📊 Employee Attrition Risk Assessment")
st.markdown("""
This production tool uses an optimized Logistic Regression engine to assess employee flight risk based on key demographic, financial, and workplace indicators.
""")
st.write("---")

st.subheader("👤 Employee Profile Input")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=65, value=35)
    monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=25000, value=5000, step=500)
    distance_from_home = st.slider("Distance From Home (miles)", min_value=1, max_value=30, value=5)

with col2:
    total_working_years = st.slider("Total Working Years", min_value=0, max_value=40, value=10)
    years_at_company = st.slider("Years at Company", min_value=0, max_value=40, value=5)
    years_in_current_role = st.slider("Years in Current Role", min_value=0, max_value=20, value=3)
    years_with_curr_manager = st.slider("Years with Current Manager", min_value=0, max_value=20, value=3)

st.write("---")
st.subheader("💼 Workplace Conditions")

col3, col4 = st.columns(2)
with col3:
    overtime = st.selectbox("Does the employee work Overtime?", options=["No", "Yes"])
with col4:
    business_travel = st.selectbox("Frequency of Business Travel", options=["Non-Travel", "Travel_Rarely", "Travel_Frequently"])

# ==========================================
# 3. DATA PROCESSING & PREDICTION PIPELINE
# ==========================================
if st.button("📊 Run Risk Assessment", type="primary"):
    
    # 1. Initialize ALL features with safe defaults
    production_row = {
        'Age': age,
        'BusinessTravel': 1 if business_travel == "Travel_Rarely" else (2 if business_travel == "Travel_Frequently" else 0),
        'DailyRate': 800,
        'DistanceFromHome': distance_from_home,
        'Education': 3,
        'EnvironmentSatisfaction': 3, 
        'Gender': 1, 
        'HourlyRate': 65,
        'JobInvolvement': 3,
        'JobLevel': 2,        
        'JobSatisfaction': 3,
        'MonthlyIncome': monthly_income,
        'MonthlyRate': 14000,
        'NumCompaniesWorked': 2,
        'OverTime': 1 if overtime == "Yes" else 0,
        'PercentSalaryHike': 15,
        'PerformanceRating': 3,
        'RelationshipSatisfaction': 3,
        'StockOptionLevel': 1,
        'TotalWorkingYears': total_working_years,
        'TrainingTimesLastYear': 2,
        'WorkLifeBalance': 3,
        'YearsAtCompany': years_at_company,
        'YearsInCurrentRole': years_in_current_role,
        'YearsSinceLastPromotion': 1,
        'YearsWithCurrManager': years_with_curr_manager,
        
        # Department dummies
        'Department_Research & Development': 1 if business_travel == "Travel_Rarely" else 0, 
        'Department_Sales': 0,
        
        # Education Field dummies
        'EducationField_Life Sciences': 1,
        'EducationField_Marketing': 0,
        'EducationField_Medical': 0,
        'EducationField_Other': 0,
        'EducationField_Technical Degree': 0,
        
        # Job Role dummies
        'JobRole_Human Resources': 0,
        'JobRole_Laboratory Technician': 0,
        'JobRole_Manager': 0,
        'JobRole_Manufacturing Director': 0,
        'JobRole_Research Director': 0,
        'JobRole_Research Scientist': 1,
        'JobRole_Sales Executive': 0,
        'JobRole_Sales Representative': 0,
        
        # Marital Status dummies
        'MaritalStatus_Married': 1,
        'MaritalStatus_Single': 0
    }
    
    
    input_df = pd.DataFrame([production_row])
    
    
    numerical_cols = [
        'Age', 
        'MonthlyIncome', 
        'DistanceFromHome', 
        'TotalWorkingYears', 
        'YearsAtCompany', 
        'YearsInCurrentRole', 
        'YearsWithCurrManager'
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
    # 4. PRODUCTION-GRADE DISPLAY
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
    
    # 2. Progress Gauge Bar
    if risk_score >= 0.50:
        st.progress(risk_score, text=f"Critical Alert Zone ({risk_percent:.1f}%)")
    else:
        st.progress(risk_score, text=f"Safe Operational Zone ({risk_percent:.1f}%)")
        
    st.write("---")
    st.write("### 📊 Key Risk Drivers Analysis")
    st.markdown("This chart isolates how the employee's active profile choices mathematically pull or push their retention probability.")

    # 3. Dynamic Visualization Data Preparation (100% Data-Driven Extraction)
    coefficients = model.coef_[0]
    feature_names = list(model.feature_names_in_)
    weight_map = dict(zip(feature_names, coefficients))

    
    ot_yes_key = next((k for k in weight_map.keys() if 'overtime' in k.lower() and 'yes' in k.lower()), None)
    ot_no_key = next((k for k in weight_map.keys() if 'overtime' in k.lower() and 'no' in k.lower()), None)

    
    if overtime == "Yes" and ot_yes_key:
        overtime_impact = weight_map[ot_yes_key]
    elif overtime == "No" and ot_no_key:
        overtime_impact = weight_map[ot_no_key]
    else:
        
        ot_single_key = next((k for k in weight_map.keys() if 'overtime' in k.lower()), None)
        if ot_single_key:
            overtime_impact = weight_map[ot_single_key] if overtime == "Yes" else 0.0
        else:
            overtime_impact = 0.0

    
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
    * This specific employee's computed probability is **{risk_score:.1%}** against an organizational baseline of **{baseline_attrition}%**.
    * {'Critical Factor: Overtime engagement increases baseline exit probabilities drastically.' if overtime == 'Yes' else 'Protective Factor: Lack of overtime drastically lowers burnout risk indicators.'}
    """)