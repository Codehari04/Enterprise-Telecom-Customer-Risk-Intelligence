# 📞 Enterprise Telecom Customer Risk Intelligence Platform (v3.0)

An enterprise-grade, end-to-end churn prediction and risk management system built with **Streamlit** and **Scikit-Learn**. This platform transforms raw telecom data into actionable business intelligence, helping retention teams prioritize high-value customers and optimize marketing spend.

---

## 🚀 Key Features

### 📊 1. Executive Dashboard
- **Real-time KPIs:** Monitor Churn Rate, Total Revenue at Risk, and Average Monthly Charges.
- **Revenue Impact:** Instant visibility into potential monthly and total revenue loss.
- **Segmentation:** Churn distribution by Gender, Seniority, and Partner status.

### 🔍 2. Advanced EDA & Insights
- **Feature Correlation:** Interactive horizontal bar charts showing exactly what drives churn.
- **Service Impact:** Analysis of how support services (Online Security, Tech Support) influence retention.
- **Driver Analysis:** Deep dives into Tenure and Monthly Charges using distribution box plots.

### 🤖 3. Automated Machine Learning
- **Multi-Model Training:** Trains and evaluates Logistic Regression, Random Forest, and Gradient Boosting.
- **Unified Pipeline:** Intelligent preprocessing (Label Encoding + Standard Scaling).
- **Performance Metrics:** Full evaluation suite including ROC-AUC, F1-Score, and Precision-Recall curves.

### 🔮 4. Individual Risk Prediction
- **Input Form:** Assess any individual customer by entering their demographics and services.
- **Probability Scoring:** Get an exact churn percentage.
- **Actionable Advice:** Dynamic retention recommendations based on risk levels (High/Medium/Low).

### 📈 5. Model Explainability (SHAP)
- **Global Importance:** See which features the model values most across the entire dataset.
- **SHAP values:** High-fidelity explanations for why the model makes specific predictions.

### ⚖️ 6. Fairness & Bias Analysis
- **Demographic Parity:** Analyze if the model is biased against specific genders or age groups.
- **Fairness Metrics:** Automated disparity calculation to ensure ethical AI deployment.

### 💼 7. Decision Intelligence
- **Cost-Benefit Analysis:** Input your own business costs for false alarms vs. missed churn.
- **Campaign Optimization:** Automatically allocates a retention budget across risk segments.
- **ROI Tracking:** Projected Revenue Recovery and Net Benefit summary.

---

## 🛠️ Technology Stack

- **Frontend:** [Streamlit](https://streamlit.io/) (Data App Framework)
- **Analysis:** Pandas, NumPy
- **Visualizations:** Plotly Express, Plotly Graph Objects, Seaborn, Matplotlib
- **Machine Learning:** Scikit-Learn (Classification, Preprocessing, Metrics)
- **Explainability:** SHAP (SHapley Additive exPlanations)
- **Deployment:** Joblib (Model Serialization)

---

## 📥 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Codehari04/Enterprise-Telecom-Risk-Intelligence.git
   cd "Enterprise Telecom Customer Risk Intelligence Platform"
   ```

2. **Install Dependencies:**
   Ensure you have Python 3.8+ installed.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

---

## 👤 Author

**Hariharan**
- **Email:** [hariharan22td0674@svcet.ac.in](mailto:hariharan22td0674@svcet.ac.in)
- **GitHub:** [Codehari04](https://github.com/Codehari04)

---

## 📝 License
This project is designed and developed for Data-Driven Decision Making. All rights reserved. © 2025 Telecom Analytics Enterprise.
