# 📊 Employee Attrition Risk Assessment Dashboard

An interactive, production-ready web application built with **Streamlit** that leverages machine learning to predict employee attrition risk. The application enables HR professionals and business stakeholders to assess the likelihood of employee turnover using key workforce, performance, and satisfaction indicators.

Live App: https://employee-attrition-risk-hl2jwjr2skzmgrbuvt5e9k.streamlit.app/
<table>
  <tr>
    <th>Homescreen</th>
    <th>Prediction</th>
  </tr>
  <tr>
    <td><img src="photos/photo1.png" alt="Home" width="400"></td>
    <td><img src="photos/photo3.png" alt="Output" width="400"></td>
  </tr>
</table>
The system transforms user-provided HR data into a production-ready feature schema, applies the same preprocessing pipeline used during model training, and generates a real-time attrition risk prediction powered by a trained **Logistic Regression model**.

Unlike traditional accuracy-focused solutions, this project prioritizes **Recall**, ensuring that potential flight-risk employees are identified as early as possible. This approach supports proactive retention strategies and minimizes the business impact associated with unexpected employee departures.

---

## 🎯 Project Objective

Employee attrition represents a significant challenge for organizations due to recruitment costs, onboarding expenses, productivity losses, and knowledge transfer gaps.

The goal of this project is to:

* Identify employees who may be at risk of leaving.
* Support proactive HR intervention strategies.
* Provide a user-friendly interface for non-technical stakeholders.
* Demonstrate an end-to-end machine learning deployment workflow.

The application bridges the gap between model development and practical business usage by transforming a trained machine learning model into an accessible decision-support tool.

---

## 🚀 Features

### 📋 Three-Tier Input Architecture

Employee information is organized into intuitive sections for improved usability:

#### Personal & Employment History

* Age
* Gender
* Marital Status
* Distance From Home
* Total Working Years
* Years at Company
* Years Since Last Promotion
* Business Travel

#### Job & Compensation Information

* Job Role
* Department
* Job Level
* Monthly Income
* Overtime Status
* Stock Option Level

#### Satisfaction & Performance Metrics

* Job Satisfaction
* Environment Satisfaction
* Relationship Satisfaction
* Work-Life Balance
* Performance Rating

---

### ⚙️ Production-Ready Data Pipeline

The dashboard replicates the exact preprocessing workflow used during model training.

Key capabilities include:

* Automatic categorical value mapping
* Dynamic one-hot encoding using `pd.get_dummies()`
* Feature alignment using saved training schema
* Missing column handling through reindexing
* Consistent numerical feature scaling
* Model-ready data transformation in real time

This ensures prediction consistency between training and deployment environments.

---

### 🤖 Machine Learning Prediction Engine

The dashboard uses a trained **Logistic Regression classifier** selected after evaluating multiple candidate models.

Models evaluated:

* Logistic Regression
* Random Forest
* XGBoost

The final model was selected because it achieved the highest **Recall**, making it the most effective model for identifying employees who ultimately left the organization.

#### Why Recall?

In employee attrition prediction:

* **True Positive:** Correctly identifies an employee who leaves.
* **False Negative:** Fails to identify an employee who leaves.

Because missing a flight-risk employee can be costly, maximizing Recall was prioritized over maximizing Accuracy alone.

Model Evaluation Summary:

| Model               | Accuracy | Recall     | Precision  | F1-Score   |
| ------------------- | -------- | ---------- | ---------- | ---------- |
| Logistic Regression | 77.55%   | **70.21%** | 39.00%     | **50.00%** |
| Random Forest       | 80.61%   | 53.19%     | 42.00%     | 46.73%     |
| XGBoost             | 81.63%   | 57.45%     | **44.00%** | **50.00%** |

As a result, Logistic Regression was selected as the production model due to its superior ability to identify employees at risk of attrition.

---

## 📈 Risk Assessment Interface

The application provides an intuitive risk analysis experience through:

### 🟢 Low Risk

Employees showing strong retention indicators and low predicted probability of attrition.

### 🟡 Moderate Risk

Employees displaying some risk factors that may warrant monitoring or engagement initiatives.

### 🔴 High Risk

Employees exhibiting multiple attrition indicators and a high likelihood of leaving the organization.

Each prediction includes:

* Attrition probability score
* Color-coded risk classification
* Visual risk progress bar
* Employee profile summary card

This allows HR teams to quickly interpret model outputs and prioritize retention efforts.

---

## 🛠️ Technology Stack

* Streamlit
* Pandas
* Scikit-learn
* Logistic Regression
* StandardScaler


---

## 📂 Project Structure

```text
employee-attrition-risk/
│
├── app/
│   ├── app.py
│   ├── attrition_model.pkl
│   ├── model_features.pkl
│   └── numerical_scaler.pkl
│
├── data/
│   └── data.csv
│
├── notebooks/
│   └── attrition_eda_ml.ipynb
│
├── photos/
│   ├── photo1.png
│   ├── photo2.png
│   └── photo3.png
│
├── .gitignore
├── README.md
└── requirements.txt
```

### Folder Details

| Path                               | Description                                               |
| ---------------------------------- | --------------------------------------------------------- |
| `app/app.py`                       | Main Streamlit dashboard application                      |
| `app/attrition_model.pkl`          | Trained Logistic Regression model                         |
| `app/model_features.pkl`           | Saved training feature schema                             |
| `app/numerical_scaler.pkl`         | StandardScaler used during model training                 |
| `data/data.csv`                    | Employee attrition dataset                                |
| `notebooks/attrition_eda_ml.ipynb` | EDA, preprocessing, model comparison, and experimentation |
| `photos/`                          | Screenshots used in project documentation                 |
| `requirements.txt`                 | Project dependencies                                      |
| `README.md`                        | Project documentation                                     |

---

## ⚡ Installation

### Clone the Repository

```bash
git clone https://github.com/anantaeeti/employee-attrition-risk.git
cd employee-attrition-risk
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app/app.py
```


## 🔍 Workflow Overview

1. User enters employee information through the dashboard.
2. Data is converted into the required machine learning format.
3. Categorical features are encoded.
4. Numerical features are standardized using the saved scaler.
5. Features are aligned with the original training schema.
6. The Logistic Regression model generates an attrition probability.
7. Results are displayed through a visual risk assessment interface.

---

## 💡 Business Value

This project demonstrates how machine learning can support human resource decision-making by:

* Improving employee retention planning
* Reducing turnover-related costs
* Identifying workforce risk patterns
* Supporting data-driven HR strategies
* Translating predictive analytics into actionable insights

---

## 📜 Disclaimer

This application is intended for educational and analytical purposes. Predictions should be used as a decision-support tool rather than a replacement for human judgment. Employee attrition is influenced by numerous organizational and personal factors that may not be fully captured by a machine learning model.
