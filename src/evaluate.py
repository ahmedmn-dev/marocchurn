"""
src/evaluate.py
───────────────
Evaluation utilities: metrics, plots, reports.

Author : Ahmed Mansof · Master Data Science · ENS Tétouan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score,
    recall_score, confusion_matrix, roc_curve, classification_report
)

DARK = '#0d1e2d'; CYAN = '#00e5ff'; PURPLE = '#7b61ff'; ORANGE = '#ff6b35'

plt.rcParams.update({
    'figure.facecolor': DARK, 'axes.facecolor': DARK,
    'axes.edgecolor': '#1a3348', 'axes.labelcolor': 'white',
    'xtick.color': 'white', 'ytick.color': 'white', 'text.color': 'white',
})


def full_report(y_true, y_pred, y_prob=None, model_name='Model') -> dict:
    """Print and return all classification metrics."""
    metrics = {
        'Accuracy':  accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall':    recall_score(y_true, y_pred),
        'F1-Score':  f1_score(y_true, y_pred),
    }
    if y_prob is not None:
        metrics['ROC-AUC'] = roc_auc_score(y_true, y_prob)

    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Report")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v*100:.2f}%")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=['No Churn', 'Churn']))
    return metrics


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix',
                          save_path=None):
    """Plot a styled confusion matrix."""
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'],
                linewidths=1, linecolor='#060d16',
                annot_kws={'size': 14, 'color': 'white'})
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.show()
    return fig


def plot_roc_curves(models_dict: dict, X_test, y_test,
                    title='ROC Curves', save_path=None):
    """Plot ROC curves for multiple models."""
    palette = [CYAN, PURPLE, ORANGE, '#00e676', '#ff4444', '#ffaa00']
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, model), color in zip(models_dict.items(), palette):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f'{name} (AUC={auc:.3f})')
    ax.plot([0, 1], [0, 1], 'w--', lw=1, alpha=0.4)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(facecolor=DARK, edgecolor='#1a3348', fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.show()
    return fig


def plot_feature_importance(model, feature_names: list,
                             top_n: int = 15, save_path=None):
    """Bar chart of feature importances."""
    fi = pd.Series(model.feature_importances_, index=feature_names)
    fi = fi.sort_values(ascending=True).tail(top_n)
    colors = [ORANGE if v > fi.quantile(0.8) else
              (CYAN if v > fi.quantile(0.5) else PURPLE) for v in fi.values]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(fi.index, fi.values, color=colors, edgecolor='none')
    for i, (feat, val) in enumerate(zip(fi.index, fi.values)):
        ax.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=8)
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.show()
    return fig
