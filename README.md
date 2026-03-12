# 📡 MarocChurn — Prédiction du Churn Télécom avec SHAP & Streamlit

> **Projet Data Science End-to-End** | Ahmed Mansof · Master Data Science · ENS Tétouan

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?style=flat-square)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red?style=flat-square)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-green?style=flat-square)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## 🎯 Objectif du projet

Un opérateur télécom perd chaque année des millions de dirhams à cause du **churn** (résiliation) de clients. Ce projet construit un système complet capable de :

1. **Prédire** quels clients vont résilier avec 96%+ de précision
2. **Expliquer** pourquoi (quels facteurs poussent chaque client à partir) via SHAP
3. **Visualiser** les résultats dans un dashboard Streamlit interactif

---

## 📁 Structure du projet

```
marocchurn/
│
├── 📓 notebooks/
│   ├── 01_EDA.ipynb              ← Analyse exploratoire complète
│   ├── 02_preprocessing.ipynb    ← Nettoyage & feature engineering
│   └── 03_modeling.ipynb         ← Entraînement & évaluation des modèles
│
├── 📦 src/
│   ├── preprocess.py             ← Pipeline de prétraitement
│   ├── train.py                  ← Entraînement des modèles
│   ├── evaluate.py               ← Métriques & visualisations
│   └── explainability.py         ← SHAP explainability
│
├── 📊 data/
│   ├── raw/
│   │   └── telco_churn.csv       ← Dataset source (Kaggle Telco)
│   └── processed/
│       └── telco_clean.csv       ← Données nettoyées
│
├── 🤖 models/
│   └── best_model_xgb.pkl        ← Meilleur modèle sauvegardé
│
├── 🖥️ app.py                     ← Dashboard Streamlit
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

**Source :** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Caractéristique | Détail |
|---|---|
| Nombre de lignes | 7 043 clients |
| Nombre de features | 21 variables |
| Target | `Churn` (Oui/Non) |
| Taux de churn | ~26% (déséquilibre de classes) |

**Variables clés :**
- `tenure` : ancienneté du client (mois)
- `MonthlyCharges` : facture mensuelle
- `Contract` : type de contrat (mensuel / 1 an / 2 ans)
- `InternetService` : type de service internet
- `TotalCharges` : total payé depuis l'inscription

---

## 🔬 Méthodologie

### Étape 1 — Analyse Exploratoire (EDA)
```
✓ Distribution des variables numériques (histogrammes, boxplots)
✓ Matrice de corrélation (heatmap Seaborn)
✓ Analyse du déséquilibre de classes
✓ Relation entre chaque feature et le churn (barplots, violin plots)
✓ Détection des valeurs manquantes et aberrantes
```

### Étape 2 — Preprocessing & Feature Engineering
```
✓ Imputation des valeurs manquantes (TotalCharges → médiane)
✓ Encodage des variables catégorielles (LabelEncoder / OHE)
✓ Normalisation MinMaxScaler sur les variables numériques
✓ Création de nouvelles features :
    - charges_ratio = MonthlyCharges / (tenure + 1)
    - is_new_customer = 1 si tenure < 6
    - high_value = 1 si MonthlyCharges > 70
✓ Gestion du déséquilibre → SMOTE oversampling
```

### Étape 3 — Modélisation & Comparaison
| Modèle | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 80.1% | 0.72 | 0.84 |
| Random Forest | 91.3% | 0.88 | 0.95 |
| Decision Tree | 85.2% | 0.81 | 0.87 |
| SVM | 82.4% | 0.75 | 0.88 |
| Gradient Boosting | 93.8% | 0.91 | 0.97 |
| **XGBoost ✓** | **96.2%** | **0.94** | **0.98** |

### Étape 4 — Optimisation
```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.9],
    'colsample_bytree': [0.7, 1.0]
}
# GridSearchCV 5-fold → meilleurs params trouvés
```

### Étape 5 — SHAP Explainability
```
✓ SHAP Values globaux (importance des features)
✓ SHAP Summary Plot (beeswarm)
✓ SHAP Force Plot (explication individuelle par client)
✓ SHAP Waterfall Plot
```

---

## Installation & Exécution en local

### Prérequis
- Python 3.10 ou supérieur
- pip ou conda

### Étape 1 — Cloner le projet
```bash
git clone https://github.com/ahmed-mansof/marocchurn.git
cd marocchurn
```

### Étape 2 — Créer un environnement virtuel
```bash
# Créer l'environnement
python -m venv venv

# Activer (Windows)
venv\Scripts\activate

# Activer (Mac/Linux)
source venv/bin/activate
```

### Étape 3 — Installer les dépendances
```bash
pip install -r requirements.txt
```

### Étape 4 — Télécharger le dataset
```
1. Aller sur : https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Télécharger WA_Fn-UseC_-Telco-Customer-Churn.csv
3. Placer le fichier dans : data/raw/telco_churn.csv
```

### Étape 5 — Lancer les notebooks (dans l'ordre)
```bash
jupyter lab
# Ouvrir et exécuter dans l'ordre :
# 01_EDA.ipynb → 02_preprocessing.ipynb → 03_modeling.ipynb
```

### Étape 6 — Lancer le dashboard Streamlit
```bash
streamlit run app.py
# Ouvrir dans le navigateur : http://localhost:8501
```

---


## 🛠 Stack Technique

```
Langage    : Python 3.10
EDA        : Pandas, NumPy, Matplotlib, Seaborn
ML         : Scikit-learn, XGBoost, imbalanced-learn (SMOTE)
XAI        : SHAP
Dashboard  : Streamlit
Notebooks  : JupyterLab
Versioning : Git, GitHub
```

---

## 👤 Auteur

**Ahmed Mansof**
- 📧 ahmedmansof.dev@gmail.com
- 💼 [LinkedIn](https://linkedin.com/in/ahmed-mansof-a73053292)
- 🐙 [GitHub](https://github.com/ahmed-mansof)

---
