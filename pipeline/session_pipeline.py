import time
import numpy as np
import tensorflow as tf
from pipeline import shared_state
from utils.constants import SESSION_LENGTH
from explainability.attention_visuals import visualize_session_attention
from tensorflow.keras.preprocessing.sequence import pad_sequences

def process_session(messages):
    start = time.time()
    if not shared_state.MODELS_LOADED:
        raise RuntimeError("Models not loaded! Call load_models() first")
    
    if not messages:
        return {
            "prediction": "non-toxic",
            "confidence": 0.0,
            "attention_plot": None,
            "toxic_messages": []
        }
    
    try:
        if len(messages) > SESSION_LENGTH:
            messages = messages[-SESSION_LENGTH:]
        
        sequences = shared_state.BILSTM_TOKENIZER.texts_to_sequences(messages)
        padded = pad_sequences(sequences, maxlen=shared_state.BILSTM_MODEL.input_shape[1], padding='post')
        inputs = np.array(padded)

        outputs = shared_state.BILSTM_MODEL(inputs, training=False)

        # If your model returns logits and attentions
        if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
            logits, attentions = outputs
        else:
            logits = outputs
            attentions = np.zeros((len(messages),))  # dummy attentions if not provided
        
        probs = tf.nn.softmax(logits).numpy()
        pred = np.argmax(probs)
        conf = probs[0][pred]

        attn_scores = np.mean(attentions, axis=1).squeeze() if attentions.ndim > 1 else attentions
        if attn_scores.ndim == 0:
            attn_scores = np.array([attn_scores])
        
        result = {
            "prediction": "toxic" if pred == 1 else "non-toxic",
            "confidence": conf,
            "attention_plot": visualize_session_attention(messages, attn_scores),
            "toxic_messages": sorted(
                [(msg, float(score)) for msg, score in zip(messages, attn_scores)],
                key=lambda x: x[1],
                reverse=True
            )[:3]
        }
        end = time.time()
        print(f"process_session took {end - start:.2f} seconds")
        return result
        
    except Exception as e:
        print(f"Error processing session: {e}")
        return {
            "prediction": "error",
            "confidence": 0.0,
            "attention_plot": None,
            "toxic_messages": []
        }
