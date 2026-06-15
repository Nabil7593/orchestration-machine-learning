# Projet fil rouge – Credit Score Classification (MLOps ESGI)

Pipeline MLOps complet de classification binaire appliqué à la prédiction de la qualité
de crédit d'un client bancaire.

---

## Etape 1 – Dataset et problematique (TP S0)

### Dataset

**Credit Score Classification** – Kaggle  
Fichier brut : `data/dataset_train.csv` (100 000 lignes, 28 colonnes)

| Attribut | Valeur |
|---|---|
| Source | [Kaggle – Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification) |
| Lignes (raw) | 100 000 |
| Colonnes brutes | 28 |
| Colonnes cibles | `Credit_Score` (Good / Standard / Poor) |

### Problematique

> **Un client bancaire a-t-il un bon score de crédit ?**

- **1 (positif)** : `Credit_Score == "Good"` – client fiable, éligible à un crédit.
- **0 (négatif)** : `Credit_Score == "Standard"` ou `"Poor"` – risque modéré ou élevé.

Prédire cette cible permet à une banque d'automatiser la pré-sélection des demandes de
crédit, en réduisant les risques de défaut de paiement.

Distribution après binarisation : ~18 % de `1` (Good) contre ~82 % de `0`.
Le dataset est **déséquilibré** → métrique principale : `roc_auc`.

### Features retenues

**Numériques (16)**

| Colonne | Description |
|---|---|
| `Age` | Âge du client |
| `Annual_Income` | Revenu annuel déclaré |
| `Monthly_Inhand_Salary` | Salaire mensuel net |
| `Num_Bank_Accounts` | Nombre de comptes bancaires |
| `Num_Credit_Card` | Nombre de cartes de crédit |
| `Interest_Rate` | Taux d'intérêt moyen |
| `Num_of_Loan` | Nombre de prêts en cours |
| `Delay_from_due_date` | Délai moyen de paiement (jours) |
| `Num_of_Delayed_Payment` | Nombre de paiements en retard |
| `Changed_Credit_Limit` | Variation de la limite de crédit |
| `Num_Credit_Inquiries` | Nombre de demandes de crédit récentes |
| `Outstanding_Debt` | Dette totale en cours |
| `Credit_Utilization_Ratio` | Taux d'utilisation du crédit |
| `Total_EMI_per_month` | Mensualités totales (EMI) |
| `Amount_invested_monthly` | Montant investi chaque mois |
| `Monthly_Balance` | Solde mensuel moyen |

**Catégorielles (4)**

| Colonne | Description |
|---|---|
| `Occupation` | Secteur d'activité |
| `Credit_Mix` | Qualité du mix crédit (Good / Standard / Bad) |
| `Payment_of_Min_Amount` | Paiement du minimum requis (Yes / No) |
| `Payment_Behaviour` | Comportement de dépense et paiement |

**Colonnes supprimées** : `ID`, `Customer_ID`, `Month`, `Name`, `SSN` (identifiants),
`Type_of_Loan` (multi-valeur complexe), `Credit_History_Age` (texte libre).
