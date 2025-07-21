import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import streamlit as st

# Set Streamlit config FIRST
st.set_page_config(page_title="Cyberbullying Detector (SVM + LR)", layout="wide")

# Path setup for imports
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# Your imports
from pipeline import shared_state
from utils.load_models import load_models, check_shared_state
from models.svm_model import custom_tokenizer  # noqa: F401
from explainability.shap_explainer import explain_lr_local, explain_lr_global

# Helpers
def gather_models() -> Dict[str, object]:
    models = {}
    if shared_state.SVM_PIPELINE is not None:
        models["SVM"] = shared_state.SVM_PIPELINE
    elif shared_state.SVM_MODEL is not None:
        models["SVM"] = shared_state.SVM_MODEL
    if shared_state.LR_PIPELINE is not None:
        models["LR"] = shared_state.LR_PIPELINE
    elif shared_state.LR_MODEL is not None:
        models["LR"] = shared_state.LR_MODEL
    if not models:
        raise RuntimeError("No SVM or Logistic-Regression models found in shared_state.")
    return models

def predict(text: str, models: Dict[str, object]) -> Dict[str, Tuple[str, float]]:
    out = {}
    for name, mdl in models.items():
        pred_idx = int(mdl.predict([text])[0])
        label = "toxic" if pred_idx else "non-toxic"
        try:
            conf = float(mdl.predict_proba([text])[0][pred_idx])
        except AttributeError:
            try:
                score = float(mdl.decision_function([text])[0])
                conf = 1 / (1 + np.exp(-score))
            except AttributeError:
                conf = 1.0
        out[name] = (label, conf)
    return out

# Main App UI
with st.spinner("Loading models..."):
    load_models()
    check_shared_state()
    MODELS = gather_models()

st.title("🛡️ Cyberbullying Detection — Classical ML + Explainable AI")

text = st.text_area("Enter a message to analyse:", height=150)
model_choice = st.radio("Choose which model to explain:", options=list(MODELS.keys()), horizontal=True)

show_lime = st.checkbox("Show LIME explanation")
show_shap = st.checkbox("Show SHAP explanation")
show_shap_global = st.checkbox("Show SHAP Global Explanation")

if st.button("Analyse"):
    if not text.strip():
        st.warning("Please enter a message first.")
        st.stop()

    preds = predict(text, MODELS)
    label, conf = preds[model_choice]

    if label == "toxic":
        st.metric("Prediction", "🚨 CYBERBULLYING", f"Confidence: {conf:.2%}", delta_color="inverse")
    else:
        st.metric("Prediction", "✅ Safe Message", f"Confidence: {conf:.2%}")

    if show_lime:
        from lime.lime_text import LimeTextExplainer
        explainer = LimeTextExplainer(class_names=["non-toxic", "toxic"])
        exp = explainer.explain_instance(text, MODELS[model_choice].predict_proba, num_features=10, labels=[1])
        st.subheader("LIME Explanation")
        st.components.v1.html(exp.as_html(), height=350, scrolling=True)

    if show_shap:
        st.subheader("SHAP Local Explanation")
        if model_choice == "LR":
            shap_fig = explain_lr_local(text, shared_state.LR_MODEL.pipeline.named_steps['logreg'])
            st.pyplot(shap_fig)
        else:
            st.warning("SHAP explanation is currently only implemented for the LR model.")

    if show_shap_global:
        st.subheader("SHAP Global Explanation")
        if model_choice == "LR":
            fig = explain_lr_global()
            st.pyplot(fig)
        else:
            st.warning("Global SHAP explanation is currently only available for the LR model.")