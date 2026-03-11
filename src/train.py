"""
src/train.py
─────────────
Script to train and save all ML models for MarocChurn.
Can be run directly from the command line.

Usage:
    python src/train.py

Author : Ahmed Mansof · Master Data Science · ENS Tétouan
"""

import pandas as pd
import numpy as np
import pickle, os, time, warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              precision_score, recall_score)
from sklearn.model_selection import GridSearchCV
import xgboost as xgb


# ── Model definitions ──────────────────────────────────────────────
MODELS = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, random_state=42),

    'Decision Tree': DecisionTreeClassifier(
        max_depth=10, random_state=42),

    'Random Forest': RandomForestClassifier(
        n_estimators=150, random_state=42, n_jobs=-1),

    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=150, random_state=42),

    'XGBoost': xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='logloss',
        random_state=42),
}

# ── GridSearchCV params for XGBoost ───────────────────────────────
XGB_PARAM_GRID = {
    'n_estimators':    [100, 200],
    'max_depth':       [4, 5, 6],
    'learning_rate':   [0.05, 0.1],
    'subsample':       [0.8, 0.9],
    'colsample_bytree':[0.8, 1.0],
}


def evaluate_model(model, X_test, y_test) -> dict:
    """Return a dict of evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        'Accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall':    round(recall_score(y_test, y_pred), 4),
        'F1-Score':  round(f1_score(y_test, y_pred), 4),
        'ROC-AUC':   round(roc_auc_score(y_test, y_prob), 4),
    }


def train_all(X_train, y_train, X_test, y_test,
              tune_best: bool = False, save_dir: str = '../models') -> dict:
    """
    Train all models and optionally tune the best one.

    Parameters
    ----------
    X_train, y_train : training data (post-SMOTE)
    X_test, y_test   : held-out test data
    tune_best        : if True, run GridSearchCV on XGBoost
    save_dir         : directory to save .pkl files

    Returns
    -------
    dict with trained models and metrics
    """
    os.makedirs(save_dir, exist_ok=True)
    trained, metrics = {}, {}

    print(f"\n{'='*60}")
    print(f"  Training {len(MODELS)} models...")
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Acc':>8} {'F1':>8} {'AUC':>8} {'Time':>8}")
    print("-" * 60)

    for name, model in MODELS.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        m = evaluate_model(model, X_test, y_test)
        metrics[name] = m
        trained[name] = model

        print(f"{name:<25} {m['Accuracy']:>8.4f} {m['F1-Score']:>8.4f} "
              f"{m['ROC-AUC']:>8.4f} {elapsed:>6.1f}s")

    # ── GridSearchCV tuning (optional) ────────────────────────────
    if tune_best:
        print(f"\n⚙️  Tuning XGBoost with GridSearchCV...")
        xgb_base = xgb.XGBClassifier(
            use_label_encoder=False, eval_metric='logloss', random_state=42)
        gs = GridSearchCV(xgb_base, XGB_PARAM_GRID,
                          cv=5, scoring='roc_auc', n_jobs=-1, verbose=0)
        t0 = time.time()
        gs.fit(X_train, y_train)
        print(f"✅ Done in {time.time()-t0:.1f}s")
        print(f"   Best params: {gs.best_params_}")
        print(f"   Best CV AUC: {gs.best_score_:.4f}")

        best_tuned = gs.best_estimator_
        m_tuned    = evaluate_model(best_tuned, X_test, y_test)
        metrics['XGBoost (Tuned)'] = m_tuned
        trained['XGBoost (Tuned)'] = best_tuned
        print(f"   Test AUC (tuned): {m_tuned['ROC-AUC']:.4f}")

    # ── Save ──────────────────────────────────────────────────────
    best_name  = max(metrics, key=lambda k: metrics[k]['ROC-AUC'])
    best_model = trained[best_name]

    with open(f'{save_dir}/best_model_xgb.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open(f'{save_dir}/all_models.pkl', 'wb') as f:
        pickle.dump(trained, f)
    with open(f'{save_dir}/all_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)

    print(f"\n✅ Best model: {best_name} (AUC={metrics[best_name]['ROC-AUC']:.4f})")
    print(f"   Saved to {save_dir}/")

    return {'trained': trained, 'metrics': metrics, 'best': best_model}


# ── CLI entry point ────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading preprocessed data...")
    try:
        X_train = pd.read_csv('../data/processed/X_train.csv')
        X_test  = pd.read_csv('../data/processed/X_test.csv')
        y_train = pd.read_csv('../data/processed/y_train.csv').squeeze()
        y_test  = pd.read_csv('../data/processed/y_test.csv').squeeze()
    except FileNotFoundError:
        print("❌ Preprocessed data not found.")
        print("   Run notebook 02_preprocessing.ipynb first, or:")
        print("   python src/preprocess.py")
        exit(1)

    result = train_all(X_train, y_train, X_test, y_test, tune_best=True)

    print("\n📊 Final Leaderboard:")
    df_m = pd.DataFrame(result['metrics']).T.sort_values('ROC-AUC', ascending=False)
    print((df_m * 100).round(2).to_string())
