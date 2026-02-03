import time
import torch
import numpy as np
from scipy.special import expit
from pipeline import shared_state
from utils.constants import XLM_CONF_THRESHOLD
from utils.text_utils import extract_toxic_words

# Lazy imports for explainability to prevent circular imports
def get_explainers():
    from explainability import attention_visuals, integrated_gradients, lime_explainer, shap_explainer
    return attention_visuals, integrated_gradients, lime_explainer, shap_explainer

class SVMWrapper:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict(self, texts):
        return self.pipeline.predict(texts)

    def predict_proba(self, texts):
        if hasattr(self.pipeline, 'predict_proba'):
            return self.pipeline.predict_proba(texts)
        else:
            print("⚠️ SVM pipeline does not support predict_proba.")
            return [[0.5, 0.5] for _ in texts]  # Return neutral prob


class LRWrapper:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict(self, texts):
        return self.pipeline.predict(texts)

    def predict_proba(self, texts):
        return self.pipeline.predict_proba(texts)

def predict_xlmr(text):
    inputs = shared_state.XLM_TOKENIZER(text, return_tensors="pt", truncation=True, max_length=128)
    device = next(shared_state.XLM_MODEL.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = shared_state.XLM_MODEL(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
    
    pred = torch.argmax(probs).item()
    conf = probs[pred].item()
    return conf, "toxic" if pred == 1 else "non-toxic"

def predict_svm(text):
    try:
        if hasattr(shared_state.SVM_MODEL, 'predict_proba'):
            proba = shared_state.SVM_MODEL.predict_proba([text])[0][1]
            pred = 1 if proba >= 0.5 else 0
            return pred, proba
        else:
            print("⚠️ SVM model does not support predict_proba.")
            return 0, 0.5
    except Exception as e:
        print(f"SVM prediction error: {str(e)}")
        return 0, 0.5

def predict_lr(text):
    try:
        proba = shared_state.LR_MODEL.predict_proba([text])[0][1]
        pred = 1 if proba >= 0.5 else 0
        return pred, proba
    except Exception as e:
        print(f"LR prediction error: {str(e)}")
        return 0, 0.5

def process_message(text):
    start = time.time()
    if not shared_state.MODELS_LOADED:
        raise RuntimeError("Models not loaded! Call load_models() first")

    # Lazy load explainers
    attention_visuals, integrated_gradients, lime_explainer, shap_explainer = get_explainers()

    xlm_conf, xlm_pred = predict_xlmr(text)
    toxic_words = extract_toxic_words(text)

    explanations = {}
    if xlm_conf >= XLM_CONF_THRESHOLD:
        try:
            explanations["attention"] = attention_visuals.visualize_xlmr_attention(text)
            explanations["ig"] = integrated_gradients.explain_xlmr_ig_embeddings(
                text, model=shared_state.XLM_MODEL, tokenizer=shared_state.XLM_TOKENIZER
            )
        except Exception as e:
            print(f"Explanation error: {e}")

        toxicity_score = xlm_conf * 100 if xlm_pred == "toxic" else (1 - xlm_conf) * 100

        result = {
            "prediction": xlm_pred,
            "confidence": xlm_conf,
            "toxicity_score": toxicity_score,
            "model": "xlmr",
            "toxic_words": toxic_words,
            "explanations": explanations
        }
    else:
        print(f"⚠️ XLM-R confidence ({xlm_conf:.2f}) below threshold — using fallback (SVM + LR)...")
        svm_pred, svm_conf = predict_svm(text)
        lr_pred, lr_conf = predict_lr(text)
        final_pred = 1 if (svm_pred + lr_pred) >= 1 else 0

        try:
            if len(text.strip()) >= 4:  # Skip very short inputs
                explanations["lime"] = lime_explainer.explain_svm(text, shared_state.SVM_MODEL)
            else:
                print("⚠️ Skipping LIME explanation — input too short.")
        except Exception as e:
            print(f"LIME explanation error: {e}")


        try:
            explanations["shap"] = shap_explainer.explain_lr_local(text, shared_state.LR_MODEL)
        except Exception as e:
            print(f"SHAP explanation error: {e}")


        # Toxicity score as max confidence for toxic class
        # Assuming svm_conf and lr_conf are probs of toxic class
        toxicity_score = max(svm_conf, lr_conf) * 100 if final_pred == 1 else (1 - max(svm_conf, lr_conf)) * 100

        result = {
            "prediction": "toxic" if final_pred == 1 else "non-toxic",
            "confidence": max(svm_conf, lr_conf),
            "toxicity_score": toxicity_score,
            "model": "ensemble",
            "toxic_words": toxic_words,
            "explanations": explanations
        }

    result["explanation_text"] = (
        f"Contains harmful words: {', '.join(toxic_words)}" 
        if toxic_words else "No harmful words detected"
    )

    print(f"Prediction completed in {time.time()-start:.2f}s")
    return result
