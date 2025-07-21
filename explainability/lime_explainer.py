from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
from pipeline import shared_state
import numpy as np

def explain_svm(text, num_features=10, num_samples=5000, class_names=None):
    try:
        if not hasattr(shared_state, 'SVM_MODEL'):
            raise ValueError("SVM model not loaded in shared_state")
            
        if class_names is None:
            class_names = ["Non-Toxic", "Toxic"]
            
        explainer = LimeTextExplainer(
            class_names=class_names,
            kernel_width=25,
            verbose=False
        )
        
        def predict_fn(texts):
            return shared_state.SVM_MODEL.predict_proba(texts)
        
        exp = explainer.explain_instance(
            text,
            predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            top_labels=2,
            distance_metric='cosine'
        )
        predicted_label = shared_state.SVM_MODEL.predict([text])[0]
        print("Predicted label:", predicted_label)
        print("Labels in explanation:", exp.local_exp.keys())

        exp_list = exp.as_list(label=predicted_label)
        print(f"LIME explanation list for label {predicted_label}:")
        for feat, weight in exp_list:
            print(f"{feat}: {weight:.4f}")

        fig, ax = plt.subplots(figsize=(10, 5))
        exp.as_pyplot_figure(label=predicted_label)
        ax.set_title("LIME Explanation", pad=20, fontsize=14)
        ax.set_xlabel("Feature Importance", fontsize=12)
        plt.tight_layout()

        return fig, exp_list
        
    except Exception as e:
        import traceback
        print("LIME Exception traceback:")
        traceback.print_exc()
        
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.text(0.5, 0.5, "Explanation failed", ha='center', va='center')
        return fig, []