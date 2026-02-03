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
        messages = messages[-3:]
        tokenizer = shared_state.BILSTM_TOKENIZER
        model = shared_state.BILSTM_MODEL

        tokenized = []
        max_len = 50

        for msg in messages:
            tokens = tokenizer.texts_to_sequences([msg])[0]
            padded_tokens = pad_sequences([tokens], maxlen=max_len, padding='post', truncating='post')[0]
            tokenized.append(padded_tokens)

        inputs = np.expand_dims(np.array(tokenized), axis=0)
        inputs_dict = {'input_layer_2': inputs}

        outputs = model(inputs_dict, training=False)

        logits = outputs.numpy()
        print("🔢 Logits:", logits)

        probs = tf.sigmoid(logits).numpy()
        print("📊 Probabilities:", probs)

        toxic_prob = probs[0][0]
        THRESHOLD = 0.5
        pred = 1 if toxic_prob > THRESHOLD else 0
        conf = toxic_prob if pred == 1 else 1 - toxic_prob

        print(f"🧠 [BiLSTM] Input messages: {messages}")
        print(f"🧠 [BiLSTM] Toxicity Probability: {toxic_prob:.4f}, Threshold: {THRESHOLD}, Prediction: {pred}")
        print(f"🧠 [BiLSTM] Confidence: {conf:.4f}")

        # Dummy attention score logic for testing
        attn_scores = [0.05 * (i+1) if pred == 1 else 0.0 for i in range(len(messages))]
        print(f"🧠 [BiLSTM] Attention Scores: {attn_scores}")

        plot = visualize_session_attention(messages, attn_scores) if any(attn_scores) else None
        print("📉 Attention plot generated" if plot else "⚠️ No plot generated")

        result = {
            "prediction": "toxic" if pred == 1 else "non-toxic",
            "confidence": conf,
            "attention_plot": plot,
            "toxic_messages": sorted(
                [(msg, float(score)) for msg, score in zip(messages, attn_scores)],
                key=lambda x: x[1],
                reverse=True
            )[:3]
        }

        print(f"✅ process_session took {time.time() - start:.2f} seconds")
        return result

    except Exception as e:
        print(f"❌ Error processing session: {e}")
        return {
            "prediction": "error",
            "confidence": 0.0,
            "attention_plot": None,
            "toxic_messages": []
        }