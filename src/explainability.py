"""
src/explainability.py
──────────────────────
SHAP-based model explainability for MarocChurn.
Provides global and local explanations.

Author : Ahmed Mansof · Master Data Science · ENS Tétouan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

DARK = '#0d1e2d'; CYAN = '#00e5ff'

plt.rcParams.update({
    'figure.facecolor': DARK, 'axes.facecolor': DARK,
    'axes.labelcolor': 'white', 'xtick.color': 'white',
    'ytick.color': 'white', 'text.color': 'white',
})


class ChurnExplainer:
    """Wraps SHAP TreeExplainer with convenience methods."""

    def __init__(self, model, X_background: pd.DataFrame):
        self.model      = model
        self.explainer  = shap.TreeExplainer(model)
        # Sample a background to keep SHAP fast
        n = min(300, len(X_background))
        self.X_bg = X_background.sample(n, random_state=42)
        self.shap_values = self.explainer.shap_values(self.X_bg)
        print(f"✅ SHAP explainer ready ({n} background samples)")

    def summary_bar(self, save_path=None):
        """Global feature importance (bar chart)."""
        fig, _ = plt.subplots(figsize=(10, 6))
        shap.summary_plot(self.shap_values, self.X_bg,
                          show=False, plot_type='bar', color=CYAN, plot_size=None)
        plt.title('SHAP — Global Feature Importance', color='white', fontsize=13)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK)
        plt.show()

    def beeswarm(self, save_path=None):
        """SHAP beeswarm — direction and magnitude."""
        fig, _ = plt.subplots(figsize=(10, 7))
        shap.summary_plot(self.shap_values, self.X_bg,
                          show=False, plot_size=None)
        plt.title('SHAP Beeswarm — Impact Direction & Magnitude',
                  color='white', fontsize=13)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK)
        plt.show()

    def explain_customer(self, row: pd.DataFrame, save_path=None):
        """
        Waterfall plot explaining one customer's prediction.

        Parameters
        ----------
        row : pd.DataFrame — single row (1 customer)
        """
        shap_row  = self.explainer.shap_values(row)
        prob      = self.model.predict_proba(row)[0][1]

        fig, _  = plt.subplots(figsize=(10, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_row[0],
                base_values=self.explainer.expected_value,
                data=row.values[0],
                feature_names=list(row.columns)
            ),
            show=False
        )
        plt.title(f'SHAP Waterfall — Churn Probability: {prob*100:.1f}%',
                  color='white', fontsize=11, pad=12)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK)
        plt.show()

        print(f"Churn probability : {prob*100:.1f}%")
        print(f"Prediction        : {'🔴 CHURN' if prob > 0.5 else '🟢 STAY'}")
        top3 = pd.Series(shap_row[0], index=row.columns).abs().sort_values(ascending=False).head(3)
        print("Top 3 factors:")
        for feat, val in top3.items():
            print(f"  {feat:<25} SHAP={shap_row[0][list(row.columns).index(feat)]:+.4f}")

        return prob, shap_row[0]

    def get_top_features(self, n: int = 10) -> pd.Series:
        """Return top-n features by mean absolute SHAP value."""
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        return pd.Series(mean_abs, index=self.X_bg.columns).sort_values(ascending=False).head(n)
