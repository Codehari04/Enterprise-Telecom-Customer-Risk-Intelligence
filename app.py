# ========================================================
# ENTERPRISE TELECOM CUSTOMER RISK INTELLIGENCE PLATFORM
# ========================================================
# Complete Production-Ready Application
# Author: Senior Data Scientist
# Version: 3.0 - Advanced Analytics Edition
# ========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import pickle
import warnings
from datetime import datetime
import io

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, f1_score, precision_score, recall_score,
    auc, ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ========================================================
# PAGE CONFIGURATION
# ========================================================
st.set_page_config(
    page_title="Telecom Risk Intelligence",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Enterprise Telecom Customer Risk Intelligence Platform v3.0"}
)

# ========================================================
# CUSTOM CSS STYLING
# ========================================================
st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary: #667eea;
        --secondary: #764ba2;
        --success: #22c55e;
        --danger: #ef4444;
        --warning: #f59e0b;
    }
    
    /* Custom styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .risk-high {
        background-image: linear-gradient(90deg, rgb(160, 222, 219),rgb(3, 165, 209));
        padding: 20px;
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.4);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 15px rgba(34, 197, 94, 0.4);
    }
    
    .header-title {
        font-size: 2.5em;
        font-weight: 800;
        background-image: linear-gradient(90deg, rgb(68, 144, 190),rgb(251, 254, 241));
        margin-bottom: 10px;
    }
    
    .info-box {
        background-image: linear-gradient(90deg, rgb(160, 222, 219),rgb(3, 165, 209));
        border-left: 4px solid #0284c7;
        padding: 15px;
        border-radius: 6px;
        margin: 15px 0;
    }
    
    .footer {
        text-align: center;
        padding: 30px;
       background-image: linear-gradient(90deg, rgb(160, 222, 219),rgb(3, 165, 209));
        border-radius: 10px;
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

# ========================================================
# UTILITY FUNCTIONS
# ========================================================

@st.cache_resource
def load_and_preprocess_data(file):
    """Load and preprocess the telecom churn dataset"""
    df = pd.read_csv(file)
    
    # Data cleaning
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    return df

def engineer_features(df):
    """Create engineered features based on business logic"""
    df = df.copy()
    
    # Prevent division by zero
    df['avg_monthly_spend'] = df['TotalCharges'] / (df['tenure'].replace(0, 1))
    
    # Service count
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['service_count'] = df[service_cols].apply(lambda x: sum(x == 'Yes'), axis=1)
    
    # Tenure buckets
    df['tenure_bucket'] = pd.cut(df['tenure'], bins=[0, 6, 24, 72], 
                                  labels=['New', 'Medium', 'Loyal'], include_lowest=True)
    
    # High-value customer
    df['high_value_customer'] = (df['TotalCharges'] > df['TotalCharges'].quantile(0.75)).astype(int)
    
    # Support risk score
    support_cols = ['OnlineSecurity', 'OnlineBackup', 'TechSupport']
    df['support_risk_score'] = df[support_cols].apply(lambda x: sum(x == 'Yes'), axis=1)
    
    # Contract stability
    df['contract_stability'] = df['Contract'].map({'Month-to-month': 1, 'One year': 2, 'Two year': 3})
    
    return df

def prepare_ml_data(df):
    """Prepare data for machine learning"""
    X = df.drop(['customerID', 'Churn'], axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    return X, y

def compute_business_cost(y_true, y_pred_proba, threshold=0.5, fn_cost=10000, fp_cost=1000):
    """Compute business cost of predictions"""
    y_pred = (y_pred_proba >= threshold).astype(int)
    fn = ((y_true == 1) & (y_pred == 0)).sum() * fn_cost
    fp = ((y_true == 0) & (y_pred == 1)).sum() * fp_cost
    return fn + fp

def get_risk_level(churn_prob):
    """Classify risk level based on churn probability"""
    if churn_prob >= 0.7:
        return "🔴 High Risk"
    elif churn_prob >= 0.4:
        return "🟠 Medium Risk"
    else:
        return "🟢 Low Risk"

def format_percentage(value):
    """Format value as percentage"""
    return f"{value*100:.2f}%"

# ========================================================
# HEADER
# ========================================================
st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1 class='header-title'>📞 Telecom Customer Risk Intelligence Platform</h1>
        <p style='color: #6b7280; font-size: 1.1em; margin-top: 0;'>
            Enterprise-Grade Churn Prediction & Risk Management System
        </p>
    </div>
""", unsafe_allow_html=True)

# ========================================================
# DATA UPLOAD & PROCESSING
# ========================================================
with st.sidebar:
    st.header("🔧 Configuration & Data")
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "📁 Upload Customer Data (CSV)",
        type=["csv"],
        help="Telecom customer churn dataset"
    )

if uploaded_file is None:
    st.info("📁 Upload customer churn data to begin analysis.")
    st.markdown("""
        ### Expected Dataset Columns:
        - **Demographics:** gender, SeniorCitizen, Partner, Dependents
        - **Services:** PhoneService, InternetService, OnlineSecurity, etc.
        - **Billing:** Contract, MonthlyCharges, TotalCharges
        - **Target:** Churn (Yes/No)
    """)
    st.stop()

# Load data
df = load_and_preprocess_data(uploaded_file)
df = engineer_features(df)

# ========================================================
# SIDEBAR NAVIGATION
# ========================================================
with st.sidebar:
    st.markdown("---")
    st.header("📑 Navigation")
    page = st.radio(
        "Select Section",
        [
            "📊 Executive Dashboard",
            "🔍 Data Analysis & EDA",
            "🎯 Churn Risk Segmentation",
            "🤖 ML Model Training",
            "🔮 Risk Prediction",
            "📈 Explainability (SHAP)",
            "⚖️ Fairness & Bias Analysis",
            "💼 Decision Intelligence"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.header("🎨 Filters & Settings")
    
    contract_filter = st.multiselect(
        "Contract Types",
        df['Contract'].unique(),
        default=df['Contract'].unique()
    )
    
    internet_filter = st.multiselect(
        "Internet Services",
        df['InternetService'].unique(),
        default=df['InternetService'].unique()
    )
    
    df_filtered = df[(df['Contract'].isin(contract_filter)) & 
                     (df['InternetService'].isin(internet_filter))]

# ========================================================
# PAGE 1: EXECUTIVE DASHBOARD
# ========================================================
if page == "📊 Executive Dashboard":
    st.subheader("📊 Executive Overview & KPIs")
    
    # Key Metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    total_customers = len(df_filtered)
    churned_customers = (df_filtered['Churn'] == 'Yes').sum()
    churn_rate = churned_customers / total_customers
    retained_customers = total_customers - churned_customers
    avg_tenure = df_filtered['tenure'].mean()
    avg_monthly_charges = df_filtered['MonthlyCharges'].mean()
    
    with col1:
        st.metric("👥 Total Customers", f"{total_customers:,}")
    with col2:
        st.metric("😢 Churned", f"{churned_customers:,}", delta=f"{format_percentage(churn_rate)}")
    with col3:
        st.metric("✅ Retained", f"{retained_customers:,}")
    with col4:
        st.metric("📊 Churn Rate", format_percentage(churn_rate), delta=f"{churned_customers} customers")
    with col5:
        st.metric("⏱️ Avg Tenure", f"{avg_tenure:.1f} months")
    with col6:
        st.metric("💰 Avg Monthly", f"${avg_monthly_charges:.2f}")
    
    st.markdown("---")
    
    # Churn Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📉 Churn Distribution")
        churn_counts = df_filtered['Churn'].value_counts()
        colors = ['#22c55e', '#ef4444']
        fig_churn = px.pie(
            values=churn_counts.values,
            names=['Retained', 'Churned'],
            title="Customer Retention vs Churn",
            color_discrete_sequence=colors,
            hole=0.4
        )
        fig_churn.update_traces(textinfo="label+percent+value")
        st.plotly_chart(fig_churn, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Churn by Contract Type")
        contract_churn = df_filtered.groupby('Contract')['Churn'].apply(
            lambda x: (x == 'Yes').sum() / len(x)
        ).sort_values(ascending=False)
        
        fig_contract = px.bar(
            x=contract_churn.values,
            y=contract_churn.index,
            orientation='h',
            title="Churn Rate by Contract Type",
            labels={"x": "Churn Rate", "y": "Contract Type"},
            color=contract_churn.values,
            color_continuous_scale="Reds",
            text=contract_churn.values
        )
        fig_contract.update_traces(texttemplate="%.1%", textposition="auto")
        st.plotly_chart(fig_contract, use_container_width=True)
    
    st.markdown("---")
    
    # Revenue at Risk Analysis
    st.markdown("### 💸 Revenue at Risk Analysis")
    
    churned_revenue = df_filtered[df_filtered['Churn'] == 'Yes']['TotalCharges'].sum()
    monthly_at_risk = df_filtered[df_filtered['Churn'] == 'Yes']['MonthlyCharges'].sum()
    potential_recovery = (df_filtered[df_filtered['Churn'] == 'No']['TotalCharges'].sum() * 
                         (df_filtered['Churn'] == 'Yes').sum() / len(df_filtered))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💰 Lost Revenue (Total)", f"${churned_revenue:,.2f}")
    with col2:
        st.metric("📈 At-Risk Monthly Revenue", f"${monthly_at_risk:,.2f}")
    with col3:
        st.metric("🎯 Potential Recovery", f"${potential_recovery:,.2f}")
    
    st.markdown("---")
    
    # Demographics Analysis
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 👨‍👩‍👧‍👦 Churn by Gender")
        gender_churn = df_filtered.groupby('gender')['Churn'].apply(
            lambda x: (x == 'Yes').sum() / len(x)
        )
        fig_gender = px.bar(
            x=gender_churn.index,
            y=gender_churn.values,
            title="Churn Rate by Gender",
            labels={"x": "Gender", "y": "Churn Rate"},
            color=gender_churn.values,
            color_continuous_scale="Oranges",
            text=gender_churn.values
        )
        fig_gender.update_traces(texttemplate="%.1%", textposition="auto")
        st.plotly_chart(fig_gender, use_container_width=True)
    
    with col2:
        st.markdown("### 👴 Churn by Senior Citizen Status")
        senior_churn = df_filtered.groupby('SeniorCitizen')['Churn'].apply(
            lambda x: (x == 'Yes').sum() / len(x)
        )
        senior_churn.index = ['Regular', 'Senior']
        fig_senior = px.bar(
            x=senior_churn.index,
            y=senior_churn.values,
            title="Churn Rate: Senior Citizens",
            labels={"x": "Status", "y": "Churn Rate"},
            color=senior_churn.values,
            color_continuous_scale="Purples",
            text=senior_churn.values
        )
        fig_senior.update_traces(texttemplate="%.1%", textposition="auto")
        st.plotly_chart(fig_senior, use_container_width=True)
    
    with col3:
        st.markdown("### 👥 Churn by Partner Status")
        partner_churn = df_filtered.groupby('Partner')['Churn'].apply(
            lambda x: (x == 'Yes').sum() / len(x)
        )
        fig_partner = px.bar(
            x=partner_churn.index,
            y=partner_churn.values,
            title="Churn Rate: Partner Status",
            labels={"x": "Partner", "y": "Churn Rate"},
            color=partner_churn.values,
            color_continuous_scale="Blues",
            text=partner_churn.values
        )
        fig_partner.update_traces(texttemplate="%.1%", textposition="auto")
        st.plotly_chart(fig_partner, use_container_width=True)

# ========================================================
# PAGE 2: DATA ANALYSIS & EDA
# ========================================================
elif page == "🔍 Data Analysis & EDA":
    st.subheader("🔍 Exploratory Data Analysis")
    
    # Dataset Overview
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📋 Total Records", len(df_filtered))
    with col2:
        st.metric("🔢 Features", len(df_filtered.columns))
    with col3:
        st.metric("❌ Missing Values", df_filtered.isnull().sum().sum())
    with col4:
        st.metric("📊 Numeric Features", len(df_filtered.select_dtypes(include=[np.number]).columns))
    with col5:
        st.metric("🏷️ Categorical Features", len(df_filtered.select_dtypes(include=['object']).columns))
    
    st.markdown("---")
    
    # Correlation Analysis
    st.markdown("### 🔥 Feature Correlation with Churn")
    
    # Prepare data for correlation
    df_corr = df_filtered.copy()
    for col in df_corr.select_dtypes(include=['object', 'category']).columns:
        if col != 'Churn':
            le = LabelEncoder()
            df_corr[col] = le.fit_transform(df_corr[col])
    df_corr['Churn'] = df_corr['Churn'].map({'Yes': 1, 'No': 0})
    
    # Calculate correlation with churn
    correlation_with_churn = df_corr.corr(numeric_only=True)['Churn'].sort_values(ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Top 10 Positive Correlations")
        top_positive = correlation_with_churn[1:11]
        fig_pos = px.bar(
            x=top_positive.values,
            y=top_positive.index,
            orientation='h',
            title="Features Increasing Churn Risk",
            labels={"x": "Correlation", "y": "Feature"},
            color=top_positive.values,
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig_pos, use_container_width=True)
    
    with col2:
        st.markdown("#### Top 10 Negative Correlations")
        bottom_negative = correlation_with_churn[-10:]
        fig_neg = px.bar(
            x=bottom_negative.values,
            y=bottom_negative.index,
            orientation='h',
            title="Features Reducing Churn Risk",
            labels={"x": "Correlation", "y": "Feature"},
            color=bottom_negative.values,
            color_continuous_scale="Greens"
        )
        st.plotly_chart(fig_neg, use_container_width=True)
    
    st.markdown("---")
    
    # Key Driver Analysis
    st.markdown("### 📊 Key Churn Drivers Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Tenure Impact")
        fig_tenure = px.box(
            df_filtered,
            x='Churn',
            y='tenure',
            title="Tenure Distribution by Churn Status",
            labels={"tenure": "Tenure (months)", "Churn": "Churn Status"},
            color='Churn',
            color_discrete_map={'Yes': '#ef4444', 'No': '#22c55e'}
        )
        st.plotly_chart(fig_tenure, use_container_width=True)
    
    with col2:
        st.markdown("#### Monthly Charges Impact")
        fig_charges = px.box(
            df_filtered,
            x='Churn',
            y='MonthlyCharges',
            title="Monthly Charges Distribution by Churn",
            labels={"MonthlyCharges": "Monthly Charges ($)", "Churn": "Churn Status"},
            color='Churn',
            color_discrete_map={'Yes': '#ef4444', 'No': '#22c55e'}
        )
        st.plotly_chart(fig_charges, use_container_width=True)
    
    st.markdown("---")
    
    # Service Analysis
    st.markdown("### 🛡️ Support Services Impact")
    
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'TechSupport', 'DeviceProtection']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Services & Churn Relationship")
        service_churn = {}
        for service in service_cols:
            churn_with = (df_filtered[df_filtered[service] == 'Yes']['Churn'] == 'Yes').sum() / len(df_filtered[df_filtered[service] == 'Yes'])
            churn_without = (df_filtered[df_filtered[service] == 'No']['Churn'] == 'Yes').sum() / len(df_filtered[df_filtered[service] == 'No'])
            service_churn[service] = {'With Service': 1 - churn_with, 'Without Service': 1 - churn_without}
        
        service_df = pd.DataFrame(service_churn).T
        fig_service = px.bar(
            service_df,
            barmode='group',
            title="Retention Rate: With vs Without Services",
            labels={"value": "Retention Rate"},
            color_discrete_sequence=["#22c55e", "#ef4444"]
        )
        st.plotly_chart(fig_service, use_container_width=True)
    
    with col2:
        st.markdown("#### Service Count Impact")
        service_count_churn = df_filtered.groupby('service_count')['Churn'].apply(
            lambda x: (x == 'Yes').sum() / len(x)
        )
        fig_count = px.line(
            x=service_count_churn.index,
            y=service_count_churn.values,
            markers=True,
            title="Churn Rate by Number of Services",
            labels={"x": "Number of Services", "y": "Churn Rate"},
            color_discrete_sequence=["#667eea"]
        )
        st.plotly_chart(fig_count, use_container_width=True)

# ========================================================
# PAGE 3: CHURN RISK SEGMENTATION
# ========================================================
elif page == "🎯 Churn Risk Segmentation":
    st.subheader("🎯 Customer Segmentation & Risk Analysis")
    
    st.markdown("""
        <div class='info-box'>
            Analyze customer segments and their churn risk profiles.
            This helps prioritize retention efforts based on segment characteristics.
        </div>
    """, unsafe_allow_html=True)
    
    # Create risk segments
    df_segment = df_filtered.copy()
    df_segment['Churn_Binary'] = (df_segment['Churn'] == 'Yes').astype(int)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔪 Contract Type Segment")
        contract_stats = df_segment.groupby('Contract').agg({
            'Churn_Binary': ['sum', 'mean', 'count'],
            'MonthlyCharges': 'mean',
            'tenure': 'mean'
        }).round(2)
        
        contract_stats.columns = ['Churned', 'Churn_Rate', 'Count', 'Avg_Charges', 'Avg_Tenure']
        st.dataframe(contract_stats, use_container_width=True)
    
    with col2:
        st.markdown("### 🌐 Internet Service Segment")
        internet_stats = df_segment.groupby('InternetService').agg({
            'Churn_Binary': ['sum', 'mean', 'count'],
            'MonthlyCharges': 'mean',
            'tenure': 'mean'
        }).round(2)
        
        internet_stats.columns = ['Churned', 'Churn_Rate', 'Count', 'Avg_Charges', 'Avg_Tenure']
        st.dataframe(internet_stats, use_container_width=True)
    
    with col3:
        st.markdown("### 💳 Payment Method Segment")
        payment_stats = df_segment.groupby('PaymentMethod').agg({
            'Churn_Binary': ['sum', 'mean', 'count'],
            'MonthlyCharges': 'mean',
            'tenure': 'mean'
        }).round(2)
        
        payment_stats.columns = ['Churned', 'Churn_Rate', 'Count', 'Avg_Charges', 'Avg_Tenure']
        st.dataframe(payment_stats, use_container_width=True)
    
    st.markdown("---")
    
    # High-Risk Segment Analysis
    st.markdown("### 🔴 High-Risk Customer Profile")
    
    high_risk = df_segment[df_segment['Churn'] == 'Yes']
    low_risk = df_segment[df_segment['Churn'] == 'No']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💰 Avg Monthly (High-Risk)", f"${high_risk['MonthlyCharges'].mean():.2f}",
                 delta=f"${high_risk['MonthlyCharges'].mean() - low_risk['MonthlyCharges'].mean():.2f}")
    with col2:
        st.metric("⏱️ Avg Tenure (High-Risk)", f"{high_risk['tenure'].mean():.1f} months",
                 delta=f"{high_risk['tenure'].mean() - low_risk['tenure'].mean():.1f} months")
    with col3:
        st.metric("🛡️ Avg Services (High-Risk)", f"{high_risk['service_count'].mean():.1f}",
                 delta=f"{high_risk['service_count'].mean() - low_risk['service_count'].mean():.1f}")
    with col4:
        st.metric("💵 Avg Total Charges (High-Risk)", f"${high_risk['TotalCharges'].mean():.2f}",
                 delta=f"${high_risk['TotalCharges'].mean() - low_risk['TotalCharges'].mean():.2f}")
    
    st.markdown("---")
    
    # Tenure & Value Segmentation Matrix
    st.markdown("### 📊 Customer Value-Tenure Matrix")
    
    df_matrix = df_segment.copy()
    df_matrix['Value'] = pd.qcut(df_matrix['TotalCharges'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
    df_matrix['Tenure_Group'] = pd.cut(df_matrix['tenure'], bins=[0, 12, 24, 72], labels=['New (<1 yr)', 'Regular (1-2 yrs)', 'Loyal (2+ yrs)'])
    
    matrix_data = df_matrix.groupby(['Tenure_Group', 'Value'])['Churn_Binary'].agg(['sum', 'mean', 'count']).reset_index()
    matrix_pivot = matrix_data.pivot_table(values='mean', index='Tenure_Group', columns='Value')
    
    fig_matrix = px.imshow(
        matrix_pivot,
        labels={"x": "Customer Value", "y": "Tenure Group", "color": "Churn Rate"},
        color_continuous_scale="RdYlGn_r",
        text_auto=".1%",
        title="Churn Rate: Customer Value × Tenure",
        aspect="auto"
    )
    st.plotly_chart(fig_matrix, use_container_width=True)

# ========================================================
# PAGE 4: ML MODEL TRAINING
# ========================================================
elif page == "🤖 ML Model Training":
    st.subheader("🤖 Machine Learning Model Development")
    
    st.markdown("""
        <div class='info-box'>
            Train and evaluate multiple ML models for churn prediction.
            Compare performance metrics and select the best model.
        </div>
    """, unsafe_allow_html=True)
    
    # Prepare data
    X, y = prepare_ml_data(df_filtered)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    st.info("⏳ Training models... Please wait.")
    
    progress_bar = st.progress(0)
    
    # Define models
    models_dict = {
        "Logistic Regression (L2)": LogisticRegression(max_iter=1000, penalty='l2', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    }
    
    results = []
    trained_models = {}
    best_auc = 0
    best_model_name = None
    best_model = None
    
    for idx, (name, model) in enumerate(models_dict.items()):
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    
    results.append({
        'Model': name,
        'ROC-AUC': auc_score,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std()
    })
    
    trained_models[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'scaler': scaler
    }
    
    if auc_score > best_auc:
        best_auc = auc_score
        best_model_name = name
        best_model = model
    
    progress_bar.progress((idx + 1) / len(models_dict))

    st.success("✅ Model training completed!")
    
    st.markdown("---")
    
    # Results Table
    st.markdown("### 📊 Model Performance Comparison")
    results_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)
    st.dataframe(results_df, use_container_width=True)
    
    st.markdown("---")
    
    # Best Model Announcement
    st.markdown(f"""
        <div class='risk-low'>
            <h3>🏆 Best Model: <strong>{best_model_name}</strong></h3>
            <p><strong>ROC-AUC:</strong> {best_auc:.4f} | 
               <strong>Precision:</strong> {results_df.iloc[0]['Precision']:.4f} |
               <strong>Recall:</strong> {results_df.iloc[0]['Recall']:.4f}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Save best model
    import os
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    st.markdown("---")
    
    # ROC Curves Comparison
    st.markdown("### 📈 ROC Curves Comparison")
    
    fig_roc = go.Figure()
    for name in trained_models.keys():
        fpr, tpr, _ = roc_curve(y_test, trained_models[name]['y_pred_proba'])
        auc_score = auc(fpr, tpr)
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f"{name} (AUC={auc_score:.3f})",
            line=dict(width=2)
        ))
    
    fig_roc.add_shape(
        type="line", line=dict(dash="dash"),
        x0=0, x1=1, y0=0, y1=1
    )
    fig_roc.update_layout(
        title="ROC Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=450
    )
    st.plotly_chart(fig_roc, use_container_width=True)
    
    st.markdown("---")
    
    # Precision-Recall Curves
    st.markdown("### 🎯 Precision-Recall Curves")
    
    fig_pr = go.Figure()
    for name in trained_models.keys():
        precision, recall, _ = precision_recall_curve(y_test, trained_models[name]['y_pred_proba'])
        ap_score = auc(recall, precision)
        fig_pr.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f"{name} (AP={ap_score:.3f})",
            line=dict(width=2)
        ))
    
    fig_pr.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=450
    )
    st.plotly_chart(fig_pr, use_container_width=True)
    
    st.markdown("---")
    
    # Confusion Matrices
    st.markdown("### 🎲 Confusion Matrices")
    
    col1, col2, col3 = st.columns(3)
    
    for idx, (name, col) in enumerate(zip(trained_models.keys(), [col1, col2, col3])):
        with col:
            cm = confusion_matrix(y_test, trained_models[name]['y_pred'])
            tn, fp, fn, tp = cm.ravel()
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['No Churn', 'Churn'],
                y=['No Churn', 'Churn'],
                text=cm,
                texttemplate="%{text}",
                colorscale='Blues'
            ))
            fig_cm.update_layout(
                title=f"{name}",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                height=400
            )
            st.plotly_chart(fig_cm, use_container_width=True)
    
    # Store in session state for other pages
    st.session_state['best_model'] = best_model
    st.session_state['best_model_name'] = best_model_name
    st.session_state['scaler'] = scaler
    st.session_state['X_test'] = X_test_scaled
    st.session_state['y_test'] = y_test
    st.session_state['trained_models'] = trained_models
    st.session_state['feature_names'] = X.columns.tolist()
elif page == "🔮 Risk Prediction":
    st.subheader("🔮 Individual Customer Risk Assessment")
    if 'best_model' not in st.session_state:
        st.error("❌ Please train the model first in 'ML Model Training' section.")
        st.stop()

    best_model = st.session_state['best_model']
    scaler = st.session_state['scaler']

    st.markdown("""
        <div class='info-box'>
            Assess individual customer churn risk and receive actionable retention recommendations.
        </div>
    """, unsafe_allow_html=True)

    # Create prediction form
    st.markdown("### 📝 Customer Information")

    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)
    
        with col1:
            st.markdown("#### 👤 Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
            partner = st.selectbox("Has Partner", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    
        with col2:
            st.markdown("#### 📱 Services")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    
        with col3:
            st.markdown("#### 💳 Billing")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 50.0)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    
        submitted = st.form_submit_button("🔍 Predict Risk", use_container_width=True)

    if submitted:
        # Create prediction input
        # Create prediction input as a single row DataFrame
        # First, ensure we have all original columns
        new_row = {
            'customerID': 'PREDICT',
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'TechSupport': tech_support,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'Contract': contract,
            'Churn': 'No'  # Dummy value for prepare_ml_data
        }
        
        # Add other original columns with default values if they are missing
        original_cols = ['MultipleLines', 'OnlineBackup', 'DeviceProtection', 'StreamingTV', 
                         'StreamingMovies', 'PaperlessBilling', 'PaymentMethod']
        for col in original_cols:
            if col not in new_row:
                new_row[col] = 'No'
        
        new_row_df = pd.DataFrame([new_row])
        
        # Combine with current filtered data to ensure consistent encoding and feature alignment
        # We use a copy of df_filtered without its engineered/target columns to avoid conflicts
        # Actually, engineer_features and prepare_ml_data handle it, but for safety:
        temp_df = pd.concat([df_filtered, new_row_df], ignore_index=True)
        
        # Re-run the preprocessing pipeline on the combined data
        # This ensures tenure_bucket, high_value_customer etc. are calculated for the new row
        temp_df = engineer_features(temp_df)
        X_combined, _ = prepare_ml_data(temp_df)
        
        # Extract the last row which is our prediction target
        X_input = X_combined.iloc[[-1]]
        
        # Ensure column order matches exactly what the scaler expects
        if hasattr(scaler, 'feature_names_in_'):
            X_input = X_input[scaler.feature_names_in_]
            
        # Scale
        X_input_scaled = scaler.transform(X_input)
        
        # Predict
        churn_prob = best_model.predict_proba(X_input_scaled)[0, 1]
        risk_level = get_risk_level(churn_prob)
    
        # Display prediction
        st.markdown("---")
        st.markdown("### 📊 Risk Assessment Results")
    
        col1, col2, col3 = st.columns(3)
    
        with col1:
            st.markdown(f"""
                <div class='risk-high' if {churn_prob > 0.7} else ('risk-medium' if {churn_prob > 0.4} else 'risk-low')>
                    <h3>Churn Probability</h3>
                    <p style='font-size: 2em; font-weight: bold;'>{format_percentage(churn_prob)}</p>
                </div>
            """, unsafe_allow_html=True)
    
        with col2:
            st.markdown(f"""
                <div class='metric-card'>
                    <h3>Risk Level</h3>
                    <p style='font-size: 1.5em;'>{risk_level}</p>
                </div>
            """, unsafe_allow_html=True)
    
        with col3:
            customer_ltv = total_charges
            at_risk_value = customer_ltv * churn_prob
            st.metric("💰 Potential Revenue at Risk", f"${at_risk_value:,.2f}",
                     delta=f"{format_percentage(churn_prob)} of ${customer_ltv:,.2f}")
    
        st.markdown("---")
    
        # Recommendations
        st.markdown("### 💡 Recommended Actions")
    
        if churn_prob >= 0.7:
            st.markdown("""
                <div class='risk-high'>
                    <h4>🔴 HIGH RISK - Immediate Action Required</h4>
                    <ul>
                        <li>📞 <strong>Immediate call center outreach</strong></li>
                        <li>🎁 <strong>Personalized retention offer</strong> (20-30% discount)</li>
                        <li>📈 <strong>Upgrade to premium services</strong></li>
                        <li>👤 <strong>Assign dedicated account manager</strong></li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
    
        elif churn_prob >= 0.4:
            st.markdown("""
                <div class='risk-medium'>
                    <h4>🟠 MEDIUM RISK - Proactive Engagement</h4>
                    <ul>
                        <li>📧 <strong>Email with service recommendations</strong></li>
                        <li>💌 <strong>Limited-time offer</strong> (10-15% discount)</li>
                        <li>🛡️ <strong>Promote support services</strong></li>
                        <li>📞 <strong>Follow-up call within 3 days</strong></li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
    
        else:
            st.markdown("""
                <div class='risk-low'>
                    <h4>🟢 LOW RISK - Maintain Relationship</h4>
                    <ul>
                        <li>✅ <strong>Regular satisfaction surveys</strong></li>
                        <li>🎉 <strong>Loyalty rewards program</strong></li>
                        <li>📱 <strong>Product update notifications</strong></li>
                        <li>👥 <strong>Community engagement opportunities</strong></li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
    
        st.markdown("---")
elif page == "📈 Explainability (SHAP)":
    st.subheader("📈 Model Explainability with SHAP")
    if 'best_model' not in st.session_state or 'feature_names' not in st.session_state:
        st.error("❌ Please train the model first in 'ML Model Training' section to enable explainability.")
        st.stop()

    st.markdown("""
        <div class='info-box'>
            Understand why the model makes specific churn predictions using SHAP (SHapley Additive exPlanations).
            SHAP values show the contribution of each feature to the final prediction.
        </div>
    """, unsafe_allow_html=True)

    best_model = st.session_state['best_model']
    X_test = st.session_state['X_test']
    feature_names = st.session_state['feature_names']

    st.markdown("### 🎯 Global Feature Importance (SHAP)")

    st.info("⏳ Computing SHAP values... This may take a moment.")

    # Create SHAP explainer
    explainer = shap.KernelExplainer(
        best_model.predict_proba,
        shap.sample(X_test, min(100, len(X_test)))
    )

    # Compute SHAP values for a sample
    shap_values = explainer.shap_values(X_test[:50])

    st.success("✅ SHAP analysis completed!")

    # Global importance
    st.markdown("#### Feature Importance (Mean Absolute SHAP)")

    # Calculate global importance robustly
    if isinstance(shap_values, list):
        # Check if list has at least two elements (for binary classification)
        # shap_values[1] is usually the 'Positive' class churn risk
        idx = 1 if len(shap_values) > 1 else 0
        mean_abs_shap = np.mean(np.abs(shap_values[idx]), axis=0)
    elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
        # Shape is (samples, features, output_classes)
        # Select the positive class (usually index 1 if binary)
        idx = 1 if shap_values.shape[2] > 1 else 0
        mean_abs_shap = np.mean(np.abs(shap_values[:, :, idx]), axis=0)
    else:
        # Fallback for standard (samples, features) 2D array
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    # Ensure it's a 1D array for DataFrame construction
    mean_abs_shap = np.array(mean_abs_shap).flatten()

    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_abs_shap
    }).sort_values('Importance', ascending=True).tail(15)

    fig_shap_imp = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Top 15 Features by SHAP Importance",
        labels={"Importance": "Mean |SHAP|", "Feature": "Feature"},
        color='Importance',
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_shap_imp, use_container_width=True)

    st.markdown("---")

    # Feature effect analysis
    st.markdown("### 📊 Feature Effect Analysis")

    selected_feature = st.selectbox("Select a feature to analyze", feature_names[:10])
    feature_idx = feature_names.index(selected_feature)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"#### {selected_feature} vs Churn Risk")
    
        feature_data = pd.DataFrame({
            'Feature Value': X_test[:100, feature_idx],
            'Churn Probability': best_model.predict_proba(X_test[:100])[:, 1]
        }).sort_values('Feature Value')
    
        fig_effect = px.scatter(
            feature_data,
            x='Feature Value',
            y='Churn Probability',
            trendline='ols',
            title=f"Partial Dependence: {selected_feature}",
            labels={"Feature Value": selected_feature, "Churn Probability": "Churn Probability"},
            color='Churn Probability',
            color_continuous_scale="RdYlGn_r",
            opacity=0.6
        )
        st.plotly_chart(fig_effect, use_container_width=True)

    with col2:
        st.markdown("#### Model Coefficients (if Linear Model)")
    
        if hasattr(best_model, 'coef_'):
            coefficients = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': best_model.coef_[0]
            }).sort_values('Coefficient', key=abs, ascending=False).head(10)
        
            fig_coef = px.bar(
                coefficients,
                x='Coefficient',
                y='Feature',
                orientation='h',
                title="Top Model Coefficients",
                color='Coefficient',
                color_continuous_scale="RdBu"
            )
            st.plotly_chart(fig_coef, use_container_width=True)
    
        st.markdown("---")
elif page == "⚖️ Fairness & Bias Analysis":
    st.subheader("⚖️ Bias & Fairness Analysis")
    st.markdown("""
        <div class='info-box'>
            Analyze model predictions across different demographic groups.
            Ensure fair and unbiased treatment of customers.
        </div>
    """, unsafe_allow_html=True)

    if 'trained_models' not in st.session_state:
        st.error("❌ Please train the model first.")
        st.stop()

    best_model = st.session_state['best_model']
    X_test = st.session_state['X_test']
    y_test = st.session_state['y_test']

    # Get predictions for analysis
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # Create analysis dataframe
    # Create analysis dataframe correctly mapped to test indices
    analysis_df = pd.DataFrame({
        'gender': df_filtered.loc[y_test.index, 'gender'].values,
        'SeniorCitizen': df_filtered.loc[y_test.index, 'SeniorCitizen'].values,
        'Churn_Probability': y_pred_proba,
        'Actual_Churn': y_test.values
    })

    st.markdown("---")
    st.markdown("### 👥 Fairness Across Demographics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Churn Prediction by Gender")
        gender_stats = analysis_df.groupby('gender').agg({
            'Churn_Probability': ['mean', 'std', 'count'],
            'Actual_Churn': 'mean'
        }).round(4)
    
        st.dataframe(gender_stats, use_container_width=True)
    
        # Visualization
        gender_pred = analysis_df.groupby('gender')['Churn_Probability'].mean()
        fig_gender_fair = px.bar(
            x=gender_pred.index,
            y=gender_pred.values,
            title="Average Churn Probability by Gender",
            labels={"x": "Gender", "y": "Avg Churn Probability"},
            color=gender_pred.values,
            color_continuous_scale="Oranges"
        )
        st.plotly_chart(fig_gender_fair, use_container_width=True)

    with col2:
        st.markdown("#### Churn Prediction by Age Group")
        age_labels = ['Regular', 'Senior']
        senior_stats = analysis_df.groupby('SeniorCitizen').agg({
            'Churn_Probability': ['mean', 'std', 'count'],
            'Actual_Churn': 'mean'
        }).round(4)
        senior_stats.index = age_labels
    
        st.dataframe(senior_stats, use_container_width=True)
    
        # Visualization
        senior_pred = analysis_df.groupby('SeniorCitizen')['Churn_Probability'].mean()
        senior_pred.index = age_labels
        fig_senior_fair = px.bar(
            x=senior_pred.index,
            y=senior_pred.values,
            title="Average Churn Probability by Age Group",
            labels={"x": "Age Group", "y": "Avg Churn Probability"},
            color=senior_pred.values,
            color_continuous_scale="Purples"
        )
        st.plotly_chart(fig_senior_fair, use_container_width=True)

    st.markdown("---")

    # Fairness Metrics
    st.markdown("### 📊 Fairness Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Demographic Parity (Gender)")
        female_churn_pred = analysis_df[analysis_df['gender'] == 'Female']['Churn_Probability'].mean()
        male_churn_pred = analysis_df[analysis_df['gender'] == 'Male']['Churn_Probability'].mean()
    
        disparity = abs(female_churn_pred - male_churn_pred)
    
        st.metric("Female Avg Churn Prob", f"{format_percentage(female_churn_pred)}")
        st.metric("Male Avg Churn Prob", f"{format_percentage(male_churn_pred)}")
        st.metric("Disparity", f"{format_percentage(disparity)}", 
                 delta="✅ Fair" if disparity < 0.05 else "⚠️ Needs Review")

    with col2:
        st.markdown("#### Demographic Parity (Age)")
        senior_churn_pred = analysis_df[analysis_df['SeniorCitizen'] == 1]['Churn_Probability'].mean()
        regular_churn_pred = analysis_df[analysis_df['SeniorCitizen'] == 0]['Churn_Probability'].mean()
    
        disparity_age = abs(senior_churn_pred - regular_churn_pred)
    
        st.metric("Senior Citizen Avg Churn Prob", f"{format_percentage(senior_churn_pred)}")
        st.metric("Regular Customer Avg Churn Prob", f"{format_percentage(regular_churn_pred)}")
        st.metric("Disparity", f"{format_percentage(disparity_age)}", 
                 delta="✅ Fair" if disparity_age < 0.05 else "⚠️ Needs Review")
elif page == "💼 Decision Intelligence":
    st.subheader("💼 Decision Intelligence & Business Strategy")
    st.markdown("""
        <div class='info-box'>
            Convert predictions into actionable business decisions.
            Optimize retention campaigns under budget constraints.
        </div>
    """, unsafe_allow_html=True)

    if 'best_model' not in st.session_state:
        st.error("❌ Please train the model first.")
        st.stop()

    st.markdown("---")
    st.markdown("### 💰 Cost-Benefit Analysis")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        fn_cost = st.number_input("Cost of Missed Churn (₹)", 10000, 50000, 10000, step=1000)
    with col2:
        fp_cost = st.number_input("Cost of False Alarm (₹)", 500, 5000, 1000, step=100)
    with col3:
        retention_success_rate = st.slider("Retention Success Rate (%)", 0, 100, 70)
    with col4:
        budget_constraint = st.number_input("Budget Constraint (₹)", 10000, 1000000, 100000, step=10000)

    st.markdown("---")

    # Risk-based prioritization
    st.markdown("### 🎯 Customer Prioritization Strategy")

    # Get predictions on full dataset
    if 'best_model' in st.session_state:
        best_model = st.session_state['best_model']
        scaler = st.session_state['scaler']
    
        X_full, y_full = prepare_ml_data(df_filtered)
        
        # Ensure column alignment
        if hasattr(scaler, 'feature_names_in_'):
            X_full = X_full[scaler.feature_names_in_]
            
        X_full_scaled = scaler.transform(X_full)
    
        churn_probs = best_model.predict_proba(X_full_scaled)[:, 1]
    
        # Create decision dataframe
        decision_df = df_filtered.copy()
        decision_df['churn_prob'] = churn_probs
        decision_df['risk_level'] = decision_df['churn_prob'].apply(get_risk_level)
        decision_df['revenue_at_risk'] = decision_df['churn_prob'] * decision_df['TotalCharges']
    
        # Prioritization logic
        decision_df['priority_score'] = (
            decision_df['churn_prob'] * 0.5 + 
            (decision_df['TotalCharges'] / decision_df['TotalCharges'].max()) * 0.3 +
            (1 - decision_df['tenure'] / decision_df['tenure'].max()) * 0.2
        )
    
        decision_df = decision_df.sort_values('priority_score', ascending=False)
    
        # Show top at-risk customers
        st.markdown("#### Top 20 At-Risk Customers (Prioritized)")
    
        top_risk = decision_df.nlargest(20, 'revenue_at_risk')[
            ['customerID', 'gender', 'tenure', 'MonthlyCharges', 'TotalCharges', 'churn_prob', 'revenue_at_risk']
        ].copy()
        top_risk['churn_prob'] = top_risk['churn_prob'].apply(format_percentage)
        top_risk['revenue_at_risk'] = top_risk['revenue_at_risk'].apply(lambda x: f"₹{x:,.0f}")
    
        st.dataframe(top_risk, use_container_width=True)
    
        st.markdown("---")
    
        # Campaign allocation
        st.markdown("### 📊 Campaign Allocation Strategy")
    
        high_risk = decision_df[decision_df['churn_prob'] >= 0.7]
        medium_risk = decision_df[(decision_df['churn_prob'] >= 0.4) & (decision_df['churn_prob'] < 0.7)]
        low_risk = decision_df[decision_df['churn_prob'] < 0.4]
    
        # Calculate campaign costs and expected outcomes
        high_risk_cost = 2000  # Premium retention offer
        medium_risk_cost = 1000  # Standard offer
        low_risk_cost = 100  # Email campaign
    
        # Budget allocation
        high_risk_allocation = min(len(high_risk), int(budget_constraint / high_risk_cost))
        remaining_budget = budget_constraint - (high_risk_allocation * high_risk_cost)
        medium_risk_allocation = min(len(medium_risk), int(remaining_budget / medium_risk_cost))
        remaining_budget -= (medium_risk_allocation * medium_risk_cost)
        low_risk_allocation = min(len(low_risk), int(remaining_budget / low_risk_cost))
    
        # Expected outcomes
        col1, col2, col3 = st.columns(3)
    
        with col1:
            st.markdown("""
                <div class='risk-high'>
                    <h4>🔴 High Risk Segment</h4>
                    <p><strong>Customers:</strong> """ + str(len(high_risk)) + """</p>
                    <p><strong>To Target:</strong> """ + str(high_risk_allocation) + """</p>
                    <p><strong>Cost/Customer:</strong> ₹""" + str(high_risk_cost) + """</p>
                    <p><strong>Expected Saves:</strong> """ + str(int(high_risk_allocation * (retention_success_rate/100) * high_risk['TotalCharges'].mean())) + """ ₹</p>
                </div>
            """, unsafe_allow_html=True)
    
        with col2:
            st.markdown("""
                <div class='risk-medium'>
                    <h4>🟠 Medium Risk Segment</h4>
                    <p><strong>Customers:</strong> """ + str(len(medium_risk)) + """</p>
                    <p><strong>To Target:</strong> """ + str(medium_risk_allocation) + """</p>
                    <p><strong>Cost/Customer:</strong> ₹""" + str(medium_risk_cost) + """</p>
                    <p><strong>Expected Saves:</strong> """ + str(int(medium_risk_allocation * (retention_success_rate/100) * medium_risk['TotalCharges'].mean())) + """ ₹</p>
                </div>
            """, unsafe_allow_html=True)
    
        with col3:
            st.markdown("""
                <div class='risk-low'>
                    <h4>🟢 Low Risk Segment</h4>
                    <p><strong>Customers:</strong> """ + str(len(low_risk)) + """</p>
                    <p><strong>To Target:</strong> """ + str(low_risk_allocation) + """</p>
                    <p><strong>Cost/Customer:</strong> ₹""" + str(low_risk_cost) + """</p>
                    <p><strong>Expected Saves:</strong> """ + str(int(low_risk_allocation * (retention_success_rate/100) * low_risk['TotalCharges'].mean())) + """ ₹</p>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("---")
    
        # ROI Summary
        st.markdown("### 💹 ROI Summary")
    
        total_cost = (high_risk_allocation * high_risk_cost) + (medium_risk_allocation * medium_risk_cost) + (low_risk_allocation * low_risk_cost)
    
        high_expected = int(high_risk_allocation * (retention_success_rate/100) * high_risk['TotalCharges'].mean())
        medium_expected = int(medium_risk_allocation * (retention_success_rate/100) * medium_risk['TotalCharges'].mean())
        low_expected = int(low_risk_allocation * (retention_success_rate/100) * low_risk['TotalCharges'].mean())
    
        total_expected_revenue = high_expected + medium_expected + low_expected
        roi = ((total_expected_revenue - total_cost) / total_cost * 100) if total_cost > 0 else 0
    
        col1, col2, col3, col4 = st.columns(4)
    
        with col1:
            st.metric("💼 Total Campaign Cost", f"₹{total_cost:,}")
        with col2:
            st.metric("💰 Expected Revenue Recovery", f"₹{total_expected_revenue:,}")
        with col3:
            st.metric("📊 Net Benefit", f"₹{total_expected_revenue - total_cost:,}")
        with col4:
            st.metric("📈 ROI %", f"{roi:.1f}%", delta="✅ Profitable" if roi > 0 else "❌ Loss")
            st.markdown("---")
    st.markdown("""
    <div class='footer'>
    <h3>📞 Telecom Customer Risk Intelligence Platform v3.0</h3>
    <p style='color: #666; margin: 10px 0;'>
    Enterprise-Grade Churn Prediction & Risk Management System
    </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; text-align: left;'>
            <div>
                <h4>📊 Capabilities</h4>
                <ul style='margin: 0; padding: 0; list-style: none; font-size: 0.9em;'>
                    <li>✅ Advanced Churn Prediction</li>
                    <li>✅ Risk Segmentation</li>
                    <li>✅ SHAP Explainability</li>
                    <li>✅ Fairness Analysis</li>
                </ul>
            </div>
            <div>
                <h4>🔧 Technologies</h4>
                <ul style='margin: 0; padding: 0; list-style: none; font-size: 0.9em;'>
                    <li>🐍 Python 3.8+</li>
                    <li>🤖 Scikit-learn</li>
                    <li>📊 Plotly & Pandas</li>
                    <li>🎯 SHAP</li>
                </ul>
            </div>
            <div>
                <h4>📈 Models</h4>
                <ul style='margin: 0; padding: 0; list-style: none; font-size: 0.9em;'>
                    <li>📉 Logistic Regression</li>
                    <li>🌳 Random Forest</li>
                    <li>⚡ Gradient Boosting</li>
                    <li>🧠 Neural Networks</li>
                </ul>
            </div>
            <div>
                <h4>📞 Support</h4>
                <ul style='margin: 0; padding: 0; list-style: none; font-size: 0.9em;'>
                    <li>📧 <a href='hariharan22td0674@svcet.ac.in' style='color: #667eea; text-decoration: none;'>Support</a></li>
                    <li>📖 <a href='update later' style='color: #667eea; text-decoration: none;'>Docs</a></li>
                    <li>🐛 <a href='update later' style='color: #667eea; text-decoration: none;'>Issues</a></li>
                    <li>⭐ <a href='https://github.com/Codehari04' style='color: #667eea; text-decoration: none;'>GitHub</a></li>
                </ul>
            </div>
        </div>
    """, unsafe_allow_html=True)
