import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Employee Attrition Prediction App", page_icon='📊', layout='centered')

@st.cache_resource
def load_assets():
    assets = {}
    assets['model'] = joblib.load('app/attrition_model.pkl')
    assets['scaler'] = joblib.load('app/numerical_scaler.pkl')
    assets['features'] = joblib.load('app/model_features.pkl')
    return assets

try:
    loaded_stuff = load_assets()
    model = loaded_stuff['model']
    scaler = loaded_stuff['scaler']
    model_features = loaded_stuff['features']
except FileNotFoundError as e:
    st.error(f"⚠️ Error: Could not find a required model file. Details: {e}")
    st.stop() 
except Exception as e:
    st.error(f"⚠️ An unexpected error occurred: {e}")
    st.stop()

st.title("📊 Employee Attrition Risk Assessment")
st.markdown("""
This production tool uses an optimized Logistic Regression engine to assess employee flight risk based on key demographic, financial, and workplace indicators.
""")
st.write("---")

user_inputs = {
    'DailyRate': 802.0, 
    'HourlyRate': 66.0,
    'MonthlyRate': 14235.5,
    'PercentSalaryHike': 14.0,
    'PerformanceRating': 3,
    'NumCompaniesWorked': 2.0,
    'RelationshipSatisfaction': 3,
    'TrainingTimesLastYear': 2,
    'Education': 3
}

tab1, tab2, tab3 = st.tabs(["📊 Personal & History", "💼 Job Role Details", "❤️ Satisfaction Survey"])

with tab1:
    st.subheader("Employee Demographics & History")
    col1, col2 = st.columns(2)
    with col1:
        user_inputs['Age'] = st.slider("Age", 18, 65, 30)
        user_inputs['DistanceFromHome'] = st.slider("Distance From Home (KM)", 1, 30, 5)
        user_inputs['TotalWorkingYears'] = st.number_input("Total Working Years", min_value=0, max_value=40, value=5)
    with col2:
        selected_gender = st.selectbox("Gender", ["Male", "Female"])
        user_inputs['Gender'] = selected_gender
        selected_marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        selected_education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"])
        user_inputs['YearsAtCompany'] = st.number_input("Years at Company", min_value=0, max_value=40, value=3)

with tab2:
    st.subheader("Job Profile & Financials")
    col3, col4 = st.columns(2)
    with col3:
        selected_dept = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
        selected_role = st.selectbox("Job Role", [
            "Research Scientist", "Laboratory Technician", "Sales Executive", 
            "Manufacturing Director", "Manager", "Research Director", 
            "Sales Representative", "Human Resources"
        ])
        user_inputs['MonthlyIncome'] = st.number_input("Monthly Income ($)", min_value=1000, max_value=25000, value=5000, step=500)
    with col4:
        selected_travel = st.selectbox("Business Travel Frequency", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
        selected_overtime = st.selectbox("Does this employee work Overtime?", ["No", "Yes"])
        user_inputs['StockOptionLevel'] = st.slider("Stock Option Level", 0, 3, 0)

with tab3:
    st.subheader("Employee Sentiment Metrics")
    st.caption("Rate the following on a scale of 1 (Low) to 4 (High):")
    col5, col6 = st.columns(2)
    with col5:
        user_inputs['JobSatisfaction'] = st.slider("Job Satisfaction", 1, 4, 3)
        user_inputs['EnvironmentSatisfaction'] = st.slider("Environment Satisfaction", 1, 4, 3)
    with col6:
        user_inputs['JobInvolvement'] = st.slider("Job Involvement", 1, 4, 3)
        user_inputs['WorkLifeBalance'] = st.slider("Work-Life Balance", 1, 4, 3)
    
    st.write("---")
    col7, col8 = st.columns(2)
    with col7:
        user_inputs['YearsInCurrentRole'] = st.number_input("Years In Current Role", min_value=0, max_value=20, value=2)
        user_inputs['YearsSinceLastPromotion'] = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=0)
    with col8:
        user_inputs['YearsWithCurrManager'] = st.number_input("Years With Current Manager", min_value=0, max_value=20, value=2)

st.write("---")

if st.button("Calculate Attrition Risk", type="primary"):
    user_inputs['Gender'] = 1 if selected_gender == "Male" else 0
    user_inputs['OverTime'] = 1 if selected_overtime == "Yes" else 0
 
    travel_map = {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}
    user_inputs['BusinessTravel'] = travel_map[selected_travel]
    
    user_inputs['MaritalStatus'] = selected_marital
    user_inputs['EducationField'] = selected_education_field
    user_inputs['Department'] = selected_dept
    user_inputs['JobRole'] = selected_role
    
    df_raw = pd.DataFrame([user_inputs])
    df_dummies = pd.get_dummies(df_raw)
    final_df = df_dummies.reindex(columns=model_features, fill_value=0)
    
    num_cols = ['Age', 'MonthlyIncome', 'DistanceFromHome', 'TotalWorkingYears', 
                'YearsAtCompany', 'YearsInCurrentRole', 'YearsWithCurrManager']
    final_df[num_cols] = scaler.transform(final_df[num_cols])
    
    risk_proba = model.predict_proba(final_df)[0][1]
    risk_percentage = risk_proba * 100
    
    st.write("### Prediction Result")
    if risk_percentage < 30:
        st.success(f"💚 **Low Attrition Risk:** This employee is likely to stay. (Risk: {risk_percentage:.1f}%)")
    elif 30 <= risk_percentage < 70:
        st.warning(f"⚠️ **Moderate Attrition Risk:** Monitor this employee. (Risk: {risk_percentage:.1f}%)")
    else:
        st.error(f"🚨 **High Attrition Risk:** High probability of attrition! (Risk: {risk_percentage:.1f}%)")
    
    st.progress(risk_proba)
    
    with st.container(border=True):
        st.write("### 📝 Employee Risk Profile Summary")
        st.caption("A consolidated view of the parameters evaluated by the Logistic Regression model.")
        st.write("")
        
        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.markdown(f"""
            #### 👤 Personal & Demographics
            * **Age:** {user_inputs['Age']} years old
            * **Marital Status:** {selected_marital}
            * **Distance From Home:** {user_inputs['DistanceFromHome']} KM
            * **Education Field:** {selected_education_field}
            """)
        with summary_col2:
            st.markdown(f"""
            #### 💼 Job & Sentiment Details
            * **Department:** {selected_dept}
            * **Job Role:** {selected_role}
            * **Monthly Income:** `${user_inputs['MonthlyIncome']:,}`
            * **Overtime Required:** `{selected_overtime}`
            """)
            
        st.write("---")
        st.markdown(f"""
        #### 📊 Key Workplace Sentiment Scores
        * **Job Satisfaction:** `{user_inputs['JobSatisfaction']}/4` | **Work-Life Balance:** `{user_inputs['WorkLifeBalance']}/4` | **Job Involvement:** `{user_inputs['JobInvolvement']}/4`
        """)

st.write("---")

with st.expander("📈 View Model Training & Performance Metrics"):
    st.markdown("""
    ## Why Logistic Regression?

    During the model evaluation phase, three machine learning algorithms were evaluated on the test set:
    * **Logistic Regression** (Selected Production Engine)
    * **XGBoost**
    * **Random Forest**

    Logistic Regression was explicitly selected because it achieved the highest **Recall (70.21%)** for the attrition class (`Left`), outperforming XGBoost (57.45%) and Random Forest (53.19%).

    ### Why Recall Matters

    In employee attrition prediction, the primary objective is to identify employees who are likely to leave the organization before attrition occurs.

    A **False Negative** occurs when the model predicts that an employee will stay, but the employee ultimately leaves. These missed cases result in unexpected workforce gaps and high replacement costs. By maximizing Recall, our model captures a much larger proportion of actual flight risks, enabling early retention interventions.

    ### Model Performance Comparison

    | Model | Recall (Class 1) | Overall Accuracy | F1-Score |
    | :--- | :---: | :---: | :---: |
    | **Logistic Regression (Ours)** | **70.21%** | **77.55%** | **50.00%** |
    | XGBoost | 57.45% | 81.63% | 50.00% |
    | Random Forest | 53.19% | 80.61% | 46.73% |

    ### Confusion Matrix Insights

    #### ✅ True Positives (Correctly Identified Attrition Risks)
    The model successfully flags 70% of the employees who intend to leave, providing HR teams with a valuable window to implement proactive retention strategies.

    #### ⚠️ False Positives (Over-Flagged Employees)
    Because our model prioritizes high recall, its precision for the attrition class is `0.39`. This means some stable employees will be flagged as risk factors. In an HR context, this is an acceptable trade-off: accidentally initiating a supportive stay-interview or check-in with a stable employee carries minimal risk compared to completely missing a valuable team member who is about to quit.
    """)