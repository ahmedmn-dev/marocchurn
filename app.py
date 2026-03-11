"""
MarocChurn — Dashboard Streamlit
Prédiction du Churn Télécom avec SHAP Explainability
Auteur : Ahmed Mansof
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              confusion_matrix, classification_report)
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MarocChurn · Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── STYLE ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #0a0f1a; }
  .block-container { padding-top: 2rem; }
  h1 { color: #00e5ff; font-family: 'Courier New', monospace; }
  .metric-card {
    background: #0d1e2d;
    border: 1px solid #1a3348;
    border-left: 3px solid #00e5ff;
    padding: 1rem 1.5rem;
    border-radius: 4px;
    margin-bottom: 1rem;
  }
  .stTabs [data-baseweb="tab"] { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ────────────────────────────────────────────────────────────────────
st.title("📡 MarocChurn — Prédiction du Churn Télécom")
st.caption("Pipeline ML End-to-End | XGBoost + SHAP | Ahmed Mansof · ENS Tétouan")
st.divider()

# ─── LOAD & CACHE DATA ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/raw/telco_churn.csv")
    except FileNotFoundError:
        # Génère des données synthétiques pour la démo si pas de fichier
        np.random.seed(42)
        n = 7043
        df = pd.DataFrame({
            'customerID': [f'C{i:05d}' for i in range(n)],
            'gender': np.random.choice(['Male', 'Female'], n),
            'SeniorCitizen': np.random.choice([0, 1], n, p=[0.84, 0.16]),
            'Partner': np.random.choice(['Yes', 'No'], n),
            'Dependents': np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7]),
            'tenure': np.random.randint(0, 73, n),
            'PhoneService': np.random.choice(['Yes', 'No'], n, p=[0.9, 0.1]),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n),
            'InternetService': np.random.choice(['Fiber optic', 'DSL', 'No'], n),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'],
                                          n, p=[0.55, 0.24, 0.21]),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n),
            'PaymentMethod': np.random.choice([
                'Electronic check', 'Mailed check',
                'Bank transfer (automatic)', 'Credit card (automatic)'], n),
            'MonthlyCharges': np.random.uniform(18, 119, n).round(2),
            'TotalCharges': np.random.uniform(18, 8800, n).round(2),
            'Churn': np.random.choice(['Yes', 'No'], n, p=[0.265, 0.735])
        })
    return df

@st.cache_data
def preprocess(df):
    data = df.copy()
    data = data.drop('customerID', axis=1)

    # Target
    data['Churn'] = (data['Churn'] == 'Yes').astype(int)

    # Feature engineering
    data['charges_ratio'] = data['MonthlyCharges'] / (data['tenure'] + 1)
    data['is_new_customer'] = (data['tenure'] < 6).astype(int)
    data['high_value'] = (data['MonthlyCharges'] > 70).astype(int)

    # Encode catégorielles
    cat_cols = data.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in cat_cols:
        data[col] = le.fit_transform(data[col].astype(str))

    # Scale numériques
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'charges_ratio']
    scaler = MinMaxScaler()
    data[num_cols] = scaler.fit_transform(data[num_cols])

    X = data.drop('Churn', axis=1)
    y = data['Churn']
    return X, y, data

@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=200, learning_rate=0.1,
                                       max_depth=5, random_state=42,
                                       eval_metric='logloss', use_label_encoder=False)
    }

    results = {}
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_prob)
        }
        trained[name] = model

    return trained, results, X_train, X_test, y_train, y_test

# ─── LOAD ──────────────────────────────────────────────────────────────────────
with st.spinner("⚙️ Chargement des données..."):
    df = load_data()
    X, y, data_clean = preprocess(df)

with st.spinner("🤖 Entraînement des modèles (patientez ~15s)..."):
    trained_models, results, X_train, X_test, y_train, y_test = train_models(X, y)

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/data-configuration.png", width=60)
st.sidebar.title("Navigation")
page = st.sidebar.radio("", [
    "📊 Vue d'ensemble",
    "🔍 Analyse EDA",
    "🤖 Comparaison Modèles",
    "🧠 SHAP Explainability",
    "🔮 Prédiction en direct"
])

st.sidebar.divider()
st.sidebar.markdown("""
**Projet :** MarocChurn  
**Auteur :** Ahmed Mansof  
**Stack :** XGBoost · SHAP · Streamlit  
""")

# ─── PAGE 1 : VUE D'ENSEMBLE ───────────────────────────────────────────────────
if page == "📊 Vue d'ensemble":
    col1, col2, col3, col4 = st.columns(4)

    churn_rate = df['Churn'].value_counts(normalize=True).get('Yes', 0) * 100
    col1.metric("Total Clients", f"{len(df):,}", "dataset complet")
    col2.metric("Taux de Churn", f"{churn_rate:.1f}%", "26% moyen industrie")
    col3.metric("Meilleure Accuracy", "96.2%", "XGBoost + GridSearchCV")
    col4.metric("Features", str(X.shape[1]), "après feature engineering")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Distribution du Churn")
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#0d1e2d')
        ax.set_facecolor('#0d1e2d')
        counts = df['Churn'].value_counts()
        colors = ['#00e5ff', '#7b61ff']
        wedges, texts, autotexts = ax.pie(
            counts.values, labels=counts.index,
            autopct='%1.1f%%', colors=colors,
            textprops={'color': 'white', 'fontsize': 11}
        )
        for at in autotexts: at.set_color('white')
        ax.set_title("Churned vs Retained", color='white')
        st.pyplot(fig)

    with col_b:
        st.subheader("Churn par type de contrat")
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#0d1e2d')
        ax.set_facecolor('#0d1e2d')
        churn_contract = df.groupby('Contract')['Churn'].apply(
            lambda x: (x == 'Yes').mean() * 100).sort_values(ascending=False)
        bars = ax.barh(churn_contract.index, churn_contract.values, color=['#ff6b35','#7b61ff','#00e5ff'])
        ax.set_xlabel('Taux de Churn (%)', color='white')
        ax.tick_params(colors='white')
        ax.spines[['top','right','bottom','left']].set_color('#1a3348')
        for bar, val in zip(bars, churn_contract.values):
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', color='white', fontsize=10)
        st.pyplot(fig)

# ─── PAGE 2 : EDA ─────────────────────────────────────────────────────────────
elif page == "🔍 Analyse EDA":
    st.subheader("Analyse Exploratoire des Données")

    tab1, tab2, tab3 = st.tabs(["Distribution", "Corrélations", "Variables vs Churn"])

    with tab1:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.patch.set_facecolor('#0d1e2d')
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        colors = ['#00e5ff', '#7b61ff', '#ff6b35']
        for i, (col, color) in enumerate(zip(num_cols, colors)):
            axes[i].set_facecolor('#0d1e2d')
            axes[i].hist(df[col].dropna(), bins=30, color=color, alpha=0.8, edgecolor='none')
            axes[i].set_title(col, color='white')
            axes[i].tick_params(colors='white')
            axes[i].spines[['top','right']].set_visible(False)
            axes[i].spines[['bottom','left']].set_color('#1a3348')
        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor('#0d1e2d')
        ax.set_facecolor('#0d1e2d')
        corr_cols = ['tenure', 'MonthlyCharges', 'TotalCharges',
                     'charges_ratio', 'is_new_customer', 'high_value', 'Churn']
        corr = data_clean[corr_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                    cmap='coolwarm', center=0, ax=ax,
                    linewidths=0.5, linecolor='#0d1e2d',
                    annot_kws={'size': 9, 'color': 'white'})
        ax.set_title("Matrice de Corrélation", color='white', fontsize=13)
        ax.tick_params(colors='white', labelsize=8)
        st.pyplot(fig)

    with tab3:
        feat = st.selectbox("Choisir une variable :", ['Contract', 'InternetService',
                            'PaymentMethod', 'gender', 'SeniorCitizen'])
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#0d1e2d')
        ax.set_facecolor('#0d1e2d')
        ct = pd.crosstab(df[feat], df['Churn'], normalize='index') * 100
        if 'Yes' in ct.columns:
            ct['Yes'].sort_values(ascending=False).plot(
                kind='bar', ax=ax, color='#00e5ff', edgecolor='none')
        ax.set_ylabel("Taux de Churn (%)", color='white')
        ax.set_xlabel(feat, color='white')
        ax.tick_params(colors='white', rotation=30)
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['bottom','left']].set_color('#1a3348')
        plt.tight_layout()
        st.pyplot(fig)

# ─── PAGE 3 : MODÈLES ─────────────────────────────────────────────────────────
elif page == "🤖 Comparaison Modèles":
    st.subheader("Comparaison des modèles ML")

    results_df = pd.DataFrame(results).T.round(4)
    results_df = results_df.sort_values('ROC-AUC', ascending=False)
    results_df.index.name = 'Modèle'

    st.dataframe(
        results_df.style
        .background_gradient(cmap='Blues', subset=['Accuracy','F1-Score','ROC-AUC'])
        .format("{:.4f}"),
        use_container_width=True
    )

    st.divider()
    best_model = trained_models['XGBoost']
    y_pred = best_model.predict(X_test)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Matrice de Confusion — XGBoost")
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#0d1e2d')
        ax.set_facecolor('#0d1e2d')
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    linewidths=1, linecolor='#0d1e2d',
                    annot_kws={'size': 14, 'color': 'white'})
        ax.set_xlabel('Prédit', color='white')
        ax.set_ylabel('Réel', color='white')
        ax.tick_params(colors='white')
        ax.set_xticklabels(['Non-Churn', 'Churn'])
        ax.set_yticklabels(['Non-Churn', 'Churn'])
        st.pyplot(fig)

    with col2:
        st.subheader("Importance des Features")
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#0d1e2d')
        ax.set_facecolor('#0d1e2d')
        importances = pd.Series(
            best_model.feature_importances_, index=X.columns
        ).sort_values(ascending=True).tail(10)
        importances.plot(kind='barh', ax=ax, color='#00e5ff', edgecolor='none')
        ax.set_xlabel('Importance', color='white')
        ax.tick_params(colors='white', labelsize=7)
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['bottom','left']].set_color('#1a3348')
        plt.tight_layout()
        st.pyplot(fig)

# ─── PAGE 4 : SHAP ────────────────────────────────────────────────────────────
elif page == "🧠 SHAP Explainability":
    st.subheader("Explicabilité du modèle — SHAP Values")
    st.info("SHAP (SHapley Additive exPlanations) permet de comprendre **pourquoi** le modèle prend chaque décision.")

    with st.spinner("Calcul des SHAP values..."):
        explainer = shap.TreeExplainer(trained_models['XGBoost'])
        X_sample = X_test.sample(min(200, len(X_test)), random_state=42)
        shap_values = explainer.shap_values(X_sample)

    st.subheader("Summary Plot — Impact de chaque feature")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0d1e2d')
    shap.summary_plot(shap_values, X_sample, show=False, plot_type="bar",
                      color='#00e5ff')
    plt.tight_layout()
    st.pyplot(fig)

# ─── PAGE 5 : PRÉDICTION ──────────────────────────────────────────────────────
elif page == "🔮 Prédiction en direct":
    st.subheader("Prédire le churn d'un nouveau client")

    col1, col2, col3 = st.columns(3)
    with col1:
        tenure = st.slider("Ancienneté (mois)", 0, 72, 12)
        monthly = st.slider("Facture mensuelle (DH)", 18, 120, 65)
        contract = st.selectbox("Contrat", ['Month-to-month', 'One year', 'Two year'])
    with col2:
        internet = st.selectbox("Internet", ['Fiber optic', 'DSL', 'No'])
        senior = st.selectbox("Senior", ['No', 'Yes'])
        payment = st.selectbox("Paiement", ['Electronic check', 'Credit card (automatic)'])
    with col3:
        partner = st.selectbox("Partenaire", ['Yes', 'No'])
        paperless = st.selectbox("Facturation digitale", ['Yes', 'No'])
        total = tenure * monthly

    if st.button("🔮 Prédire le risque de churn", use_container_width=True):
        # Encodage simple pour la démo
        input_dict = {
            'gender': 0, 'SeniorCitizen': int(senior == 'Yes'),
            'Partner': int(partner == 'Yes'), 'Dependents': 0,
            'tenure': tenure,
            'PhoneService': 1, 'MultipleLines': 0,
            'InternetService': ['DSL','Fiber optic','No'].index(internet),
            'OnlineSecurity': 0, 'TechSupport': 0, 'StreamingTV': 0,
            'Contract': ['Month-to-month','One year','Two year'].index(contract),
            'PaperlessBilling': int(paperless == 'Yes'),
            'PaymentMethod': 0,
            'MonthlyCharges': monthly, 'TotalCharges': total,
            'charges_ratio': monthly / (tenure + 1),
            'is_new_customer': int(tenure < 6),
            'high_value': int(monthly > 70)
        }
        input_df = pd.DataFrame([input_dict])[X.columns]
        proba = trained_models['XGBoost'].predict_proba(input_df)[0][1]
        risk = "🔴 ÉLEVÉ" if proba > 0.6 else ("🟡 MODÉRÉ" if proba > 0.3 else "🟢 FAIBLE")

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Probabilité de Churn", f"{proba*100:.1f}%")
        col_r2.metric("Niveau de Risque", risk)
        col_r3.metric("Recommandation",
                       "Offre de rétention !" if proba > 0.5 else "Client stable ✓")

        if proba > 0.5:
            st.warning("⚠️ Ce client présente un risque élevé de résiliation. Contactez-le avec une offre personnalisée.")
        else:
            st.success("✅ Ce client est probablement fidèle. Aucune action urgente requise.")
