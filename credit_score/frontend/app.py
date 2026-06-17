"""Frontend Streamlit - Credit Score Classifier.

Seance 14 bis - TP Streamlit
    Appelle l'API FastAPI /predict et affiche le resultat.
    Lancement local :
        PYTHONPATH=src streamlit run frontend/app.py
    En docker compose : API_URL=http://api:8000 est injecte automatiquement.
"""
from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Score Classifier",
    page_icon="🏦",
    layout="wide",
)
st.title("🏦 Credit Score Classifier")
st.caption("Prédit si un client a un bon score de crédit (Good = 1 / Standard-Poor = 0)")

api_url = st.text_input("URL de l'API", value=API_URL)

predict_tab, history_tab = st.tabs(["Prédiction", "Historique"])

# ---------------------------------------------------------------------------
# Onglet Prédiction
# ---------------------------------------------------------------------------
with predict_tab:
    st.subheader("Informations du client")

    with st.form("predict_form"):

        # S14bis-1 : champs adaptés au dataset Credit Score
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Profil financier**")
            Age = st.number_input("Âge", min_value=18, max_value=100, value=35)
            Annual_Income = st.number_input(
                "Revenu annuel (€)", min_value=0.0, value=50000.0, step=1000.0
            )
            Monthly_Inhand_Salary = st.number_input(
                "Salaire mensuel net (€)", min_value=0.0, value=4000.0, step=100.0
            )
            Outstanding_Debt = st.number_input(
                "Dette en cours (€)", min_value=0.0, value=1500.0, step=100.0
            )
            Credit_Utilization_Ratio = st.number_input(
                "Taux d'utilisation crédit (%)", min_value=0.0, max_value=100.0, value=25.0
            )
            Monthly_Balance = st.number_input(
                "Solde mensuel (€)", value=500.0, step=50.0
            )

        with col2:
            st.markdown("**Comportement de crédit**")
            Num_Bank_Accounts = st.number_input(
                "Nb comptes bancaires", min_value=0, max_value=20, value=3
            )
            Num_Credit_Card = st.number_input(
                "Nb cartes de crédit", min_value=0, max_value=20, value=4
            )
            Interest_Rate = st.number_input(
                "Taux d'intérêt (%)", min_value=0.0, max_value=50.0, value=15.0
            )
            Num_of_Loan = st.number_input(
                "Nb de prêts en cours", min_value=0, max_value=20, value=2
            )
            Num_Credit_Inquiries = st.number_input(
                "Nb demandes de crédit", min_value=0, max_value=20, value=2
            )
            Changed_Credit_Limit = st.number_input(
                "Variation limite crédit (%)", value=2.5, step=0.5
            )

        with col3:
            st.markdown("**Historique de paiement**")
            Delay_from_due_date = st.number_input(
                "Retard moyen (jours)", min_value=0, max_value=365, value=5
            )
            Num_of_Delayed_Payment = st.number_input(
                "Nb paiements en retard", min_value=0, max_value=50, value=3
            )
            Total_EMI_per_month = st.number_input(
                "Mensualités EMI (€)", min_value=0.0, value=150.0, step=10.0
            )
            Amount_invested_monthly = st.number_input(
                "Investissement mensuel (€)", min_value=0.0, value=200.0, step=10.0
            )

            st.markdown("**Profil**")
            Occupation = st.selectbox("Profession", [
                "Accountant", "Architect", "Developer", "Doctor", "Engineer",
                "Entrepreneur", "Journalist", "Lawyer", "Manager", "Mechanic",
                "Media_Manager", "Musician", "Scientist", "Teacher", "Writer",
            ])
            Credit_Mix = st.selectbox("Mix de crédit", ["Standard", "Good", "Bad"])
            Payment_of_Min_Amount = st.selectbox(
                "Paiement du minimum requis", ["No", "Yes", "NM"]
            )
            Payment_Behaviour = st.selectbox("Comportement de paiement", [
                "High_spent_Small_value_payments",
                "High_spent_Medium_value_payments",
                "High_spent_Large_value_payments",
                "Low_spent_Small_value_payments",
                "Low_spent_Medium_value_payments",
                "Low_spent_Large_value_payments",
            ])

        submitted = st.form_submit_button("🔍 Prédire", use_container_width=True)

    # S14bis-2 : payload avec les memes cles que le schema Features de l'API
    if submitted:
        payload = {
            "Age": float(Age),
            "Annual_Income": Annual_Income,
            "Monthly_Inhand_Salary": Monthly_Inhand_Salary,
            "Num_Bank_Accounts": float(Num_Bank_Accounts),
            "Num_Credit_Card": float(Num_Credit_Card),
            "Interest_Rate": Interest_Rate,
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
            "Occupation": Occupation,
            "Credit_Mix": Credit_Mix,
            "Payment_of_Min_Amount": Payment_of_Min_Amount,
            "Payment_Behaviour": Payment_Behaviour,
        }

        try:
            response = httpx.post(f"{api_url}/predict", json=payload, timeout=10.0)
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError as exc:
            st.error(f"Appel à l'API impossible : {exc}")
        else:
            # S14bis-3 : affichage du résultat
            prediction = result["prediction"]
            probability = result["probability"]

            st.divider()

            res_col1, res_col2 = st.columns(2)

            with res_col1:
                if prediction == 1:
                    st.success("✅ **Good** — Bon score de crédit")
                else:
                    st.error("❌ **Standard / Poor** — Score de crédit insuffisant")

            with res_col2:
                st.metric(
                    label="Probabilité d'être Good",
                    value=f"{probability:.1%}",
                )
                st.progress(probability)

# ---------------------------------------------------------------------------
# Onglet Historique (bonus)
# ---------------------------------------------------------------------------
with history_tab:
    st.subheader("Historique des prévisions")
    st.info(
        "Aucun journal de prévisions : "
        "ajoutez un endpoint /predictions à l'API (bonus S14bis-4)."
    )
