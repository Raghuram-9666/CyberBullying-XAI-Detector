import shap
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from pipeline import shared_state
from sklearn.utils import check_array

def explain_lr_global(n_features=20):
    """Global SHAP explanation using loaded pipeline"""
    try:
        if not hasattr(shared_state, 'LR_MODEL'):
            raise ValueError("LR model not loaded")
            
        # Get pre-computed training sample from shared_state
        X_train_sample = shared_state.X_TRAIN_SAMPLE
        X_train_sample = check_array(X_train_sample, accept_sparse=True)
        
        if X_train_sample.shape[0] > 500:
            X_train_sample = X_train_sample[np.random.choice(X_train_sample.shape[0], 500, replace=False)]
        
        explainer = shap.LinearExplainer(
            shared_state.LR_MODEL.pipeline.named_steps['logreg'], 
            X_train_sample
        )
        shap_values = explainer.shap_values(X_train_sample)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(
            shap_values, 
            X_train_sample, 
            feature_names=shared_state.VECTORIZER.get_feature_names_out(),
            plot_type="bar",
            max_display=n_features,
            show=False
        )
        ax.set_title("Global Feature Importance", pad=20, fontsize=14)
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"SHAP Global Error: {str(e)}")
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.text(0.5, 0.5, "Global explanation failed", ha='center', va='center')
        return fig

# ──────────────────────────────────────────────────────────────────────────────
# Helper for fallback plots (leave this exactly as you already have it)
def _generate_fallback_plot(message):
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.text(0.5, 0.5, message, ha='center', va='center')
    return fig


def explain_lr_local(text: str, model, background_size: int = 100):
    """
    Local SHAP explanation for a single message.

    • Vectorises the input text with shared_state.VECTORIZER
    • Subsamples shared_state.X_TRAIN_SAMPLE to `background_size`
    • Converts sparse matrices to dense for plotting safety
    • Handles binary‑classifier list‑of‑arrays output
    • Always returns a matplotlib Figure
    """
    try:
        vect = shared_state.VECTORIZER
        if vect is None:
            raise ValueError("Vectoriser missing from shared_state")

        X = vect.transform([text])
        if X.shape[1] == 0:
            return _generate_fallback_plot("Vectorised input is empty")

        bg = shared_state.X_TRAIN_SAMPLE
        if bg is None:
            raise ValueError("Background sample missing from shared_state")

        if bg.shape[0] > background_size:
            rows = np.random.choice(bg.shape[0], background_size, replace=False)
            bg = bg[rows]

        if hasattr(bg, "toarray"):
            bg = bg.toarray()
        if hasattr(X, "toarray"):
            X_plot = X.toarray()
        else:
            X_plot = X

        explainer = shap.LinearExplainer(
            model,
            bg,
            feature_perturbation="interventional"
        )
        shap_vals = explainer.shap_values(X)

        # BONUS: print shap_vals info for debugging
        print(f"Type of shap_vals: {type(shap_vals)}")
        print(f"Shape of shap_vals: {np.shape(shap_vals)}")

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        feature_names = vect.get_feature_names_out()

        fig = shap.force_plot(
            explainer.expected_value if hasattr(explainer, "expected_value") else explainer.base_values,
            shap_vals[0],
            X_plot[0],
            feature_names=feature_names,
            matplotlib=True,
            show=False,
        )
        fig.set_size_inches(12, 3)
        plt.tight_layout()
        return fig

    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        st.error(f"❌ SHAP Local Error:\n```\n{err_msg}\n```")
        return _generate_fallback_plot("Local explanation failed")