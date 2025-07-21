import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load saved model and tokenizer
def load_model_and_tokenizer():
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved_models', 'bilstm_model')
    model_path = os.path.join(base_dir, 'bilstm_sequence_model.keras')
    tokenizer_path = os.path.join(base_dir, 'tokenizer.pkl')

    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

# Preprocess and pad text inputs to shape (1, 3, 50)
def preprocess_texts(texts, tokenizer, max_len=50):
    tokenized_seq = []
    for msg in texts:
        tokens = tokenizer.texts_to_sequences([msg])[0]
        padded = pad_sequences([tokens], maxlen=max_len, padding='post', truncating='post')[0]
        tokenized_seq.append(padded)

    # Shape: (sequence_length, max_len) → expand dims to (1, sequence_length, max_len)
    return np.expand_dims(np.array(tokenized_seq), axis=0)

def main():
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    print("Ready! Enter messages one by one.")

    session_msgs = []
    max_len = 50
    sequence_length = 3

    while True:
        msg = input("User message (type 'exit' to quit): ")
        if msg.lower() == 'exit':
            print("Exiting demo.")
            break

        session_msgs.append(msg)

        if len(session_msgs) >= sequence_length:
            input_texts = session_msgs[-sequence_length:]  # Last 3 messages
            input_data = preprocess_texts(input_texts, tokenizer, max_len)

            preds = model.predict(input_data)
            avg_pred = preds[0][0]  # Since shape is (1, 1)

            label = "Bullying/ Toxic" if avg_pred > 0.5 else "Non-bullying/ Non-toxic"
            print(f"Session prediction (last {sequence_length} msgs): {label} (score: {avg_pred:.3f})")

if __name__ == "__main__":
    main()
