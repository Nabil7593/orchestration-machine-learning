"""Frontend Streamlit - Credit Score Classifier."""
from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
EC2_IP = os.environ.get("EC2_IP", "35.181.53.42")

st.set_page_config(
    page_title="Credit Score AI — Nabil SARKER",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS custom
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); min-height: 100vh; }

    .hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
    }
    .hero h1 {
        font-size: 3em;
        font-weight: 900;
        color: white;
        margin: 0;
        letter-spacing: -1px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    .hero h2 {
        font-size: 1.3em;
        color: rgba(255,255,255,0.85);
        margin: 10px 0 0 0;
        font-weight: 300;
    }
    .hero .subtitle {
        font-size: 0.95em;
        color: rgba(255,255,255,0.65);
        margin-top: 8px;
    }

    .card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        backdrop-filter: blur(10px);
    }
    .card h3 { color: #a78bfa; font-size: 1.1em; margin: 0 0 16px 0; font-weight: 600; }

    .result-good {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        color: white;
        font-size: 1.5em;
        font-weight: 700;
        box-shadow: 0 10px 30px rgba(56, 239, 125, 0.3);
    }
    .result-bad {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        color: white;
        font-size: 1.5em;
        font-weight: 700;
        box-shadow: 0 10px 30px rgba(245, 87, 108, 0.3);
    }

    .link-card {
        background: rgba(102, 126, 234, 0.15);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 14px 18px;
        margin: 8px 0;
        text-decoration: none;
        display: block;
        transition: all 0.2s;
    }
    .link-card:hover { background: rgba(102, 126, 234, 0.3); }
    .link-card span { font-size: 1.3em; }
    .link-card p { color: #c4b5fd; margin: 4px 0 0 0; font-size: 0.85em; }

    .badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.75em;
        font-weight: 600;
        margin: 4px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 1.1em !important;
        padding: 14px !important;
        width: 100% !important;
        transition: all 0.3s !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
    }

    .section-title {
        color: #a78bfa;
        font-size: 0.85em;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="hero">
    <h1>🏦 Credit Score AI</h1>
    <h2>Nabil SARKER — ESGI MLOps 2025</h2>
    <div class="subtitle">Système de classification du risque crédit par Machine Learning</div>
    <div style="margin-top: 16px;">
        <span class="badge">FastAPI</span>
        <span class="badge">Scikit-learn</span>
        <span class="badge">MLflow</span>
        <span class="badge">Docker</span>
        <span class="badge">AWS EC2</span>
        <span class="badge">Airflow</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — liens
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <div style="font-size:2.5em;">🏦</div>
        <div style="color:#a78bfa; font-weight:700; font-size:1.1em;">Nabil SARKER</div>
        <div style="color:#6b7280; font-size:0.8em;">ESGI MLOps · 2025</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-title">🔗 Services déployés</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <a class="link-card" href="http://{EC2_IP}:8000/docs" target="_blank">
        <span>⚡</span> <strong style="color:white;">API FastAPI</strong>
        <p>Documentation interactive Swagger</p>
    </a>
    <a class="link-card" href="http://{EC2_IP}:5001" target="_blank">
        <span>📊</span> <strong style="color:white;">MLflow UI</strong>
        <p>Expériences & métriques du modèle</p>
    </a>
    <a class="link-card" href="http://{EC2_IP}:8080" target="_blank">
        <span>🔄</span> <strong style="color:white;">Airflow UI</strong>
        <p>Orchestration du ré-entraînement</p>
    </a>
    <a class="link-card" href="https://github.com/Nabil7593/orchestration-machine-learning" target="_blank">
        <span>🐙</span> <strong style="color:white;">GitHub Repo</strong>
        <p>Code source complet du projet</p>
    </a>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-title">⚙️ Configuration</div>', unsafe_allow_html=True)
    api_url = st.text_input("URL de l'API", value=API_URL, label_visibility="collapsed",
                             placeholder="http://...")

    st.divider()
    st.markdown("""
    <div style="color:#4b5563; font-size:0.75em; text-align:center; padding:10px 0;">
        Projet MLOps ESGI · Paris<br>
        Classification binaire — 100k clients<br>
        ROC-AUC ≈ 0.85
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
predict_tab, info_tab = st.tabs(["🔍 Prédiction", "📈 À propos du modèle"])

# ---------------------------------------------------------------------------
# Onglet Prédiction
# ---------------------------------------------------------------------------
with predict_tab:
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="section-title">💰 Profil financier</div>', unsafe_allow_html=True)
            Age = st.number_input("Âge", min_value=18, max_value=100, value=35)
            Annual_Income = st.number_input("Revenu annuel (€)", min_value=0.0, value=50000.0, step=1000.0)
            Monthly_Inhand_Salary = st.number_input("Salaire mensuel net (€)", min_value=0.0, value=4000.0, step=100.0)
            Outstanding_Debt = st.number_input("Dette en cours (€)", min_value=0.0, value=1500.0, step=100.0)
            Credit_Utilization_Ratio = st.number_input("Taux d'utilisation crédit (%)", min_value=0.0, max_value=100.0, value=25.0)
            Monthly_Balance = st.number_input("Solde mensuel (€)", value=500.0, step=50.0)

        with col2:
            st.markdown('<div class="section-title">💳 Comportement crédit</div>', unsafe_allow_html=True)
            Num_Bank_Accounts = st.number_input("Nb comptes bancaires", min_value=0, max_value=20, value=3)
            Num_Credit_Card = st.number_input("Nb cartes de crédit", min_value=0, max_value=20, value=4)
            Interest_Rate = st.number_input("Taux d'intérêt (%)", min_value=0.0, max_value=50.0, value=15.0)
            Num_of_Loan = st.number_input("Nb de prêts en cours", min_value=0, max_value=20, value=2)
            Num_Credit_Inquiries = st.number_input("Nb demandes de crédit", min_value=0, max_value=20, value=2)
            Changed_Credit_Limit = st.number_input("Variation limite crédit (%)", value=2.5, step=0.5)

        with col3:
            st.markdown('<div class="section-title">📅 Historique paiement</div>', unsafe_allow_html=True)
            Delay_from_due_date = st.number_input("Retard moyen (jours)", min_value=0, max_value=365, value=5)
            Num_of_Delayed_Payment = st.number_input("Nb paiements en retard", min_value=0, max_value=50, value=3)
            Total_EMI_per_month = st.number_input("Mensualités EMI (€)", min_value=0.0, value=150.0, step=10.0)
            Amount_invested_monthly = st.number_input("Investissement mensuel (€)", min_value=0.0, value=200.0, step=10.0)

            st.markdown('<div class="section-title" style="margin-top:12px;">👤 Profil</div>', unsafe_allow_html=True)
            Occupation = st.selectbox("Profession", [
                "Accountant", "Architect", "Developer", "Doctor", "Engineer",
                "Entrepreneur", "Journalist", "Lawyer", "Manager", "Mechanic",
                "Media_Manager", "Musician", "Scientist", "Teacher", "Writer",
            ])
            Credit_Mix = st.selectbox("Mix de crédit", ["Standard", "Good", "Bad"])
            Payment_of_Min_Amount = st.selectbox("Paiement du minimum requis", ["No", "Yes", "NM"])
            Payment_Behaviour = st.selectbox("Comportement de paiement", [
                "High_spent_Small_value_payments",
                "High_spent_Medium_value_payments",
                "High_spent_Large_value_payments",
                "Low_spent_Small_value_payments",
                "Low_spent_Medium_value_payments",
                "Low_spent_Large_value_payments",
            ])

        submitted = st.form_submit_button("🔍 Analyser le profil crédit", use_container_width=True)

    if submitted:
        payload = {
            "Age": float(Age), "Annual_Income": Annual_Income,
            "Monthly_Inhand_Salary": Monthly_Inhand_Salary,
            "Num_Bank_Accounts": float(Num_Bank_Accounts),
            "Num_Credit_Card": float(Num_Credit_Card), "Interest_Rate": Interest_Rate,
            "Num_of_Loan": float(Num_of_Loan),
            "Delay_from_due_date": float(Delay_from_due_date),
            "Num_of_Delayed_Payment": float(Num_of_Delayed_Payment),
            "Changed_Credit_Limit": Changed_Credit_Limit,
            "Num_Credit_Inquiries": float(Num_Credit_Inquiries),
            "Outstanding_Debt": Outstanding_Debt,
            "Credit_Utilization_Ratio": Credit_Utilization_Ratio,
            "Total_EMI_per_month": Total_EMI_per_month,
            "Amount_invested_monthly": Amount_invested_monthly,
            "Monthly_Balance": Monthly_Balance,
            "Occupation": Occupation, "Credit_Mix": Credit_Mix,
            "Payment_of_Min_Amount": Payment_of_Min_Amount,
            "Payment_Behaviour": Payment_Behaviour,
        }

        with st.spinner("Analyse en cours..."):
            try:
                response = httpx.post(f"{api_url}/predict", json=payload, timeout=10.0)
                response.raise_for_status()
                result = response.json()
            except httpx.HTTPError as exc:
                st.error(f"❌ Erreur API : {exc}")
            else:
                prediction = result["prediction"]
                probability = result["probability"]

                st.markdown("---")
                col_r1, col_r2, col_r3 = st.columns([2, 1, 1])

                with col_r1:
                    if prediction == 1:
                        st.markdown("""
                        <div class="result-good">
                            ✅ GOOD<br>
                            <span style="font-size:0.6em; font-weight:400;">Bon profil crédit</span>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="result-bad">
                            ❌ STANDARD / POOR<br>
                            <span style="font-size:0.6em; font-weight:400;">Profil risqué</span>
                        </div>""", unsafe_allow_html=True)

                with col_r2:
                    st.metric("Probabilité Good", f"{probability:.1%}")

                with col_r3:
                    st.metric("Score risque", f"{1 - probability:.1%}")

                st.progress(probability, text=f"Confiance du modèle : {probability:.1%}")

# ---------------------------------------------------------------------------
# Onglet À propos
# ---------------------------------------------------------------------------
with info_tab:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div class="card">
            <h3>🎯 Objectif</h3>
            <p style="color:#d1d5db;">Classification binaire du score de crédit :<br>
            <strong style="color:#a78bfa;">Good (1)</strong> vs <strong style="color:#f87171;">Standard/Poor (0)</strong></p>
            <p style="color:#9ca3af; font-size:0.9em;">Dataset Kaggle · 100 000 clients · 20 features</p>
        </div>
        <div class="card">
            <h3>📊 Performances</h3>
            <p style="color:#d1d5db;">ROC-AUC : <strong style="color:#34d399;">≈ 0.85</strong></p>
            <p style="color:#d1d5db;">F1-score : <strong style="color:#34d399;">≈ 0.70</strong></p>
            <p style="color:#9ca3af; font-size:0.9em;">Modèle : LogisticRegression (baseline)</p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="card">
            <h3>🏗️ Architecture</h3>
            <p style="color:#d1d5db;">
            ⚡ <strong>FastAPI</strong> — API REST d'inférence<br>
            📊 <strong>MLflow</strong> — Tracking des expériences<br>
            🔄 <strong>Airflow</strong> — Ré-entraînement planifié<br>
            🐳 <strong>Docker Compose</strong> — Stack conteneurisée<br>
            ☁️ <strong>AWS EC2</strong> — Déploiement cloud<br>
            🚀 <strong>GitHub Actions</strong> — CI/CD automatisé
            </p>
        </div>
        <div class="card">
            <h3>👨‍💻 Auteur</h3>
            <p style="color:#d1d5db;">
            <strong style="color:#a78bfa; font-size:1.2em;">Nabil SARKER</strong><br>
            Étudiant MLOps · ESGI Paris · 2025<br><br>
            <a href="https://github.com/Nabil7593/orchestration-machine-learning"
               target="_blank" style="color:#667eea;">
               🐙 github.com/Nabil7593/orchestration-machine-learning
            </a>
            </p>
        </div>
        """, unsafe_allow_html=True)
