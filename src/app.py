import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
from retention_ai import get_retention_strategy

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load Model
model = pickle.load(open("model.pkl", "rb"))

# =========================
# TITLE
# =========================

st.markdown(
    """
    <h1 style='text-align:center; color:#4CAF50;'>
    📊 Customer Churn Prediction Dashboard
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# =========================
# KPI CARDS
# =========================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "👥 Total Customers",
        "7043"
    )

with col2:
    st.metric(
        "📉 Churn Rate",
        "26.5%"
    )

with col3:
    st.metric(
        "🎯 Model Accuracy",
        "78.7%"
    )

st.markdown("---")

# =========================
# PIE CHART
# =========================

st.subheader("📊 Customer Distribution")

churn_data = pd.DataFrame({
    "Status": ["Stayed", "Churned"],
    "Count": [5174, 1869]
})

fig = px.pie(
    churn_data,
    names="Status",
    values="Count",
    hole=0.5,
    title="Customer Churn Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================
# SIDEBAR
# =========================

st.sidebar.title("📌 Project Information")

st.sidebar.success("Machine Learning Project")

st.sidebar.write("""
### Project Details

**Project Name:** Customer Churn Prediction

**Model:** Logistic Regression

**Accuracy:** 78.7%

### Technologies Used

- Python
- Pandas
- Scikit-Learn
- Streamlit
- Plotly


""")

# =========================
# CUSTOMER FORM
# =========================

st.markdown(
    """
    <h2 style='color:#2196F3;'>
    👤 Customer Information
    </h2>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12
    )

with col2:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=50.0
    )

contract = st.selectbox(
    "Contract Type",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

# =========================
# PREDICTION
# =========================

if st.button("🚀 Predict Churn"):

    contract_map = {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2
    }

    contract_value = contract_map[contract]

    input_data = pd.DataFrame(
        [[
            tenure,
            monthly_charges,
            contract_value
        ]],
        columns=[
            "tenure",
            "MonthlyCharges",
            "Contract"
        ]
    )

    risk = model.predict_proba(input_data)[0][1]

    st.markdown(
        """
        <h2 style='color:#FF9800;'>
        📈 Prediction Result
        </h2>
        """,
        unsafe_allow_html=True
    )

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            "Risk %",
            f"{risk*100:.2f}%"
        )

    with m2:
        st.metric(
            "Tenure",
            tenure
        )

    with m3:
        st.metric(
            "Monthly Charges",
            monthly_charges
        )

    st.progress(float(risk))

    if risk >= 0.8:
        st.error(
            f"🔴 High Risk Customer ({risk*100:.2f}%)"
        )

    elif risk >= 0.5:
        st.warning(
            f"🟡 Medium Risk Customer ({risk*100:.2f}%)"
        )

    else:
        st.success(
            f"🟢 Low Risk Customer ({risk*100:.2f}%)"
        )

    recommendation = get_retention_strategy(risk)

    st.markdown(
        """
        <h2 style='color:#9C27B0;'>
        🎯 Retention Recommendation
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.info(recommendation)

# =========================
# ABOUT PROJECT
# =========================

st.markdown("---")

st.markdown("""
## 📖 About Project

This project predicts customer churn using Machine Learning
and provides intelligent retention recommendations.

### Project Workflow

Customer Data
➡ Data Cleaning
➡ Feature Engineering
➡ Logistic Regression
➡ Churn Prediction
➡ Retention Strategy
➡ Dashboard

### Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Plotly
""")