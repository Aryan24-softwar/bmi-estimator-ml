"""
Streamlit Web App — BMI Estimator
Author : Vishwakarma Aryan Santosh
Project: ML Minor Project
"""

import os, pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# -- Page Config ---------------------------------------------------------------
st.set_page_config(
    page_title="BMI Estimator — ML Project",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Load Models ---------------------------------------------------------------
BASE   = os.path.dirname(os.path.abspath(__file__))
MODEL  = os.path.join(BASE, "models")
DATA   = os.path.join(BASE, "data", "bmi_dataset.csv")
PLOTS  = os.path.join(BASE, "plots")


@st.cache_resource
def load_models():
    reg  = pickle.load(open(os.path.join(MODEL, "bmi_regressor.pkl"),  "rb"))
    clf  = pickle.load(open(os.path.join(MODEL, "bmi_classifier.pkl"), "rb"))
    scl  = pickle.load(open(os.path.join(MODEL, "scaler.pkl"),         "rb"))
    le   = pickle.load(open(os.path.join(MODEL, "label_encoder.pkl"),  "rb"))
    ce   = pickle.load(open(os.path.join(MODEL, "cat_encoder.pkl"),    "rb"))
    return reg, clf, scl, le, ce


@st.cache_data
def load_data():
    return pd.read_csv(DATA,encoding='latin-1')


try:
    reg_model, clf_model, scaler, le, cat_enc = load_models()
    df = load_data()
    models_ready = True
except FileNotFoundError:
    models_ready = False

# -- Custom CSS ----------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #3949ab 100%);
        padding: 2rem; border-radius: 15px; margin-bottom: 1.5rem; text-align: center;
    }
    .main-header h1 { color: white; font-size: 2.4rem; margin: 0; }
    .main-header p  { color: #c5cae9; font-size: 1rem; margin: 0.3rem 0 0; }
    .bmi-card {
        border-radius: 15px; padding: 1.5rem; text-align: center;
        margin: 0.5rem 0; border: 2px solid transparent; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .bmi-value { font-size: 3.5rem; font-weight: 900; }
    .category-badge {
        display: inline-block; padding: 0.4rem 1.2rem; border-radius: 25px;
        font-size: 1.1rem; font-weight: 700; margin-top: 0.5rem;
    }
    .metric-box {
        background: white; border-radius: 10px; padding: 1rem;
        text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .stButton > button {
        background: linear-gradient(135deg, #1a237e, #3949ab);
        color: white; border: none; padding: 0.7rem 2rem;
        border-radius: 10px; font-size: 1rem; font-weight: 700;
        width: 100%; transition: all 0.3s;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(57,73,171,0.4); }
    .footer { text-align: center; color: #9e9e9e; padding: 1rem; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# -- Header ---------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>?? BMI Estimator</h1>
    <p>Machine Learning Minor Project &nbsp;|&nbsp; Author: <strong>Vishwakarma Aryan Santosh</strong></p>
</div>
""", unsafe_allow_html=True)

if not models_ready:
    st.error("?? Models not found! Please run **train_model.py** first.")
    st.code("python generate_dataset.py\npython train_model.py")
    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ?? Enter Your Details")
    st.markdown("---")

    gender = st.selectbox("?? Gender", ["Male", "Female"])

    unit = st.radio("?? Unit System", ["Metric (cm, kg)", "Imperial (ft/in, lbs)"])

    if unit == "Metric (cm, kg)":
        height = st.slider("?? Height (cm)", 130, 220, 170)
        weight = st.slider("?? Weight (kg)", 30, 180, 70)
    else:
        col1, col2 = st.columns(2)
        with col1:
            feet = st.number_input("Feet", 4, 7, 5)
        with col2:
            inches = st.number_input("Inches", 0, 11, 7)
        height = (feet * 30.48) + (inches * 2.54)
        weight_lbs = st.slider("?? Weight (lbs)", 66, 400, 154)
        weight = weight_lbs * 0.453592

    st.markdown("---")
    predict_btn = st.button("?? Estimate BMI")

    st.markdown("---")
    st.markdown("### ?? BMI Reference")
    st.markdown("""
| Range       | Category      |
|-------------|---------------|
| < 18.5      | ?? Underweight |
| 18.5 – 24.9 | ?? Normal weight|
| 25.0 – 29.9 | ?? Overweight  |
| = 30.0      | ?? Obese       |
""")

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["?? Prediction", "?? Data Insights", "?? Model Info"])

# -- TAB 1: Prediction ---------------------------------------------------------
with tab1:
    gender_enc = le.transform([gender])[0]
    X_input = scaler.transform([[gender_enc, height, weight]])
    bmi_pred = reg_model.predict(X_input)[0]
    cat_enc_pred = clf_model.predict(X_input)[0]
    category = cat_enc.inverse_transform([cat_enc_pred])[0]

    CAT_CONFIG = {
        "Underweight":  {"color": "#1565C0", "bg": "#E3F2FD", "icon": "??", "advice": "Consider a nutritionist-recommended diet to gain healthy weight."},
        "Normal weight":{"color": "#2E7D32", "bg": "#E8F5E9", "icon": "??", "advice": "Great job! Maintain your current lifestyle and stay active."},
        "Overweight":   {"color": "#E65100", "bg": "#FFF3E0", "icon": "??", "advice": "Consider increasing physical activity and adopting a balanced diet."},
        "Obese":        {"color": "#B71C1C", "bg": "#FFEBEE", "icon": "??", "advice": "Consult a healthcare professional for a structured weight management plan."},
    }
    cfg = CAT_CONFIG.get(category, CAT_CONFIG["Normal weight"])

    if predict_btn or True:   # always show live result
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.markdown(f"""
            <div class="bmi-card" style="background:{cfg['bg']}; border-color:{cfg['color']};">
                <div style="font-size:1rem; color:#555; margin-bottom:0.3rem;">Your Estimated BMI</div>
                <div class="bmi-value" style="color:{cfg['color']};">{bmi_pred:.1f}</div>
                <div class="category-badge" style="background:{cfg['color']}; color:white;">
                    {cfg['icon']} {category}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("?? Gender",     gender)
        col_b.metric("?? Height",     f"{height:.1f} cm")
        col_c.metric("?? Weight",     f"{weight:.1f} kg")
        col_d.metric("?? BMI",        f"{bmi_pred:.2f}")

        st.markdown("---")
        st.info(f"?? **Health Tip:** {cfg['advice']}")

        # BMI Gauge
        st.markdown("### ?? BMI Scale")
        fig, ax = plt.subplots(figsize=(10, 1.8))
        ranges = [(0, 18.5, "#1565C0"), (18.5, 25, "#2E7D32"), (25, 30, "#E65100"), (30, 45, "#B71C1C")]
        labels = ["Underweight", "Normal weight", "Overweight", "Obese"]
        for i, (lo, hi, col) in enumerate(ranges):
            ax.barh(0, hi - lo, left=lo, height=0.6, color=col, alpha=0.85)
            ax.text((lo + hi) / 2, 0, labels[i], ha="center", va="center", color="white",
                    fontsize=9, fontweight="bold")
        ax.axvline(bmi_pred, color="black", linewidth=3, zorder=5)
        ax.annotate(f"  {bmi_pred:.1f}", xy=(bmi_pred, 0.3),
                    fontsize=12, fontweight="bold", color="black")
        ax.set_xlim(10, 45); ax.set_ylim(-0.5, 0.5)
        ax.axis("off")
        ax.set_facecolor("#f8f9fa")
        fig.patch.set_facecolor("#f8f9fa")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# -- TAB 2: Data Insights ------------------------------------------------------
with tab2:
    st.markdown("### ?? Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Avg BMI",       f"{df['BMI'].mean():.2f}")
    c3.metric("Min BMI",       f"{df['BMI'].min():.2f}")
    c4.metric("Max BMI",       f"{df['BMI'].max():.2f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### BMI Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df["BMI"], kde=True, color="#3949AB", ax=ax, bins=40)
        ax.set_xlabel("BMI"); ax.set_ylabel("Count")
        ax.axvline(bmi_pred, color="red", linestyle="--", lw=2, label=f"You: {bmi_pred:.1f}")
        ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("#### BMI Category Distribution")
        order  = ["Underweight", "Normal weight", "Overweight", "Obese"]
        colors = ["#1565C0", "#2E7D32", "#E65100", "#B71C1C"]
        counts = df["BMI_Category"].value_counts().reindex(order)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(order, counts.values, color=colors)
        ax.set_ylabel("Count"); plt.xticks(rotation=15, ha="right")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Height vs Weight (BMI Category)")
        palette = {"Underweight": "#1565C0", "Normal weight": "#2E7D32",
                   "Overweight": "#E65100", "Obese": "#B71C1C"}
        fig, ax = plt.subplots(figsize=(6, 4))
        for cat, grp in df.groupby("BMI_Category"):
            ax.scatter(grp["Height_cm"], grp["Weight_kg"],
                       label=cat, alpha=0.35, s=10, color=palette.get(cat,"gray"))
        ax.scatter([height], [weight], color="black", s=150, zorder=10,
                   marker="*", label="You")
        ax.set_xlabel("Height (cm)"); ax.set_ylabel("Weight (kg)")
        ax.legend(fontsize=8); plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("#### BMI by Gender")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x="Gender", y="BMI",
                    palette={"Male": "#42A5F5", "Female": "#EC407A"}, ax=ax)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("#### Sample Dataset")
    st.dataframe(df.head(20), use_container_width=True)

# -- TAB 3: Model Info ---------------------------------------------------------
with tab3:
    st.markdown("### ?? ML Models Used")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Regression (BMI Value Estimation)")
        reg_info = {
            "Algorithm":   type(reg_model).__name__,
            "Task":        "Predict exact BMI value",
            "Features":    "Gender, Height, Weight",
            "Output":      "Continuous BMI value",
        }
        for k, v in reg_info.items():
            st.write(f"**{k}:** {v}")

    with col2:
        st.markdown("#### Classification (BMI Category)")
        clf_info = {
            "Algorithm":   type(clf_model).__name__,
            "Task":        "Predict BMI Category",
            "Features":    "Gender, Height, Weight",
            "Output":      "Underweight / Normal / Overweight / Obese",
        }
        for k, v in clf_info.items():
            st.write(f"**{k}:** {v}")

    st.markdown("---")
    st.markdown("### ?? BMI Formula")
    st.latex(r"\text{BMI} = \frac{\text{Weight (kg)}}{\text{Height (m)}^2}")

    st.markdown("---")
    st.markdown("### ?? Pipeline")
    st.markdown("""
    ```
    Raw Input (Gender, Height, Weight)
         ?
    Label Encoding (Gender: Male=1, Female=0)
         ?
    Feature Scaling (StandardScaler)
         ?
    +------------------------------------------+
    ¦  BMI Regressor      ¦  BMI Classifier    ¦
    ¦  (Continuous Value) ¦  (Category Label)  ¦
    +------------------------------------------+
         ?                        ?
    BMI Value              BMI Category
    ```
    """)

    plots_exist = os.path.exists(PLOTS)
    if plots_exist:
        st.markdown("---")
        st.markdown("### ?? Training Visualisations")
        cols = st.columns(2)
        plot_files = [f for f in os.listdir(PLOTS) if f.endswith(".png")]
        plot_files.sort()
        for i, pf in enumerate(plot_files):
            cols[i % 2].image(os.path.join(PLOTS, pf), use_container_width=True)

# -- Footer --------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div class="footer">
    ?? BMI Estimator &nbsp;|&nbsp; ML Minor Project &nbsp;|&nbsp;
    <strong>Vishwakarma Aryan Santosh</strong> &nbsp;|&nbsp; 2025–26
</div>
""", unsafe_allow_html=True)
