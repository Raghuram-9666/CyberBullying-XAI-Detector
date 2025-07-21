"""
cli_main_svm_lr.py
A terminal‑based helper that loads *only* the SVM and Logistic‑Regression
models saved by your pipeline (using the variables already present in
pipeline.shared_state) and prints their individual predictions.
Run it in the same virtual‑env / conda‑env as the rest of the project:

    python cli_main_svm_lr.py
"""
from __future__ import annotations

import sys
from typing import Dict, Tuple

from pipeline import shared_state
from utils.load_models import load_models, check_shared_state

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

def gather_classical_models() -> Dict[str, object]:
    """Return a dict with the SVM and LR estimators that were loaded by load_models().

    The project stores them in shared_state as either *_PIPELINE (preferred: a
    full scikit‑learn Pipeline containing vectoriser + classifier) or *_MODEL
    (just the fitted estimator). This helper unifies that into a simple dict
    keyed by an easy‑to‑type short name ("svm", "lr").
    """
    models: Dict[str, object] = {}

    # SVM first
    if shared_state.SVM_PIPELINE is not None:
        models["svm"] = shared_state.SVM_PIPELINE
    elif shared_state.SVM_MODEL is not None:
        models["svm"] = shared_state.SVM_MODEL

    # Logistic Regression next
    if shared_state.LR_PIPELINE is not None:
        models["lr"] = shared_state.LR_PIPELINE
    elif shared_state.LR_MODEL is not None:
        models["lr"] = shared_state.LR_MODEL

    if not models:
        raise RuntimeError(
            "Neither SVM nor Logistic‑Regression models were found in shared_state.\n"
            "Make sure load_models() is populating SVM_PIPELINE / LR_PIPELINE or the *_MODEL variables."
        )

    return models


def predict_with_models(text: str, models: Dict[str, object]) -> Dict[str, Tuple[str, float]]:
    """Return {model_name: (label, confidence)} for each model supplied."""
    results: Dict[str, Tuple[str, float]] = {}

    for name, mdl in models.items():
        # Binary classifiers: assume class index 1 is the "toxic" class
        pred_idx = int(mdl.predict([text])[0])
        label = "toxic" if pred_idx else "non‑toxic"

        # Try calibrated probability; otherwise approximate from decision_function or fallback
        try:
            conf = float(mdl.predict_proba([text])[0][pred_idx])
        except AttributeError:
            # If decision_function exists, squash through sigmoid for a pseudo‑probability
            try:
                import numpy as np  # numpy is already a project dependency
                score = float(mdl.decision_function([text])[0])
                conf = 1.0 / (1.0 + np.exp(-score))
            except AttributeError:
                conf = 1.0  # last‑ditch fallback when calibration not available

        results[name] = (label, conf)

    return results

# ----------------------------------------------------------------------------
# CLI loop
# ----------------------------------------------------------------------------

def cli_loop() -> None:
    """Interactive REPL that prints each model’s verdict for every user message."""
    # 1. Load *all* project models so the shared_state variables get populated.
    load_models()
    check_shared_state()

    # 2. Pull out just SVM + LR
    models = gather_classical_models()

    print("\n🚀 Cyberbullying Detection CLI (SVM + LR)")
    print("Type a message and press Enter. Type 'exit' or press Ctrl‑C to quit.\n")

    while True:
        try:
            text = input("Enter message: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting CLI.")
            break

        if text.lower() in {"exit", "quit"}:
            print("Exiting CLI.")
            break
        if not text:
            print("Please enter some text.")
            continue

        try:
            preds = predict_with_models(text, models)

            # Pretty‑print per‑model results
            for model_name, (label, conf) in preds.items():
                icon = "🚨" if label == "toxic" else "✅"
                print(f"[{model_name.upper():>3}] {icon} {label.upper()} (confidence {conf:.2%})")

            print("-" * 50)

        except Exception as exc:
            print(f"\n⚠️  Error during prediction: {exc}")
            print("-" * 50)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        cli_loop()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user")