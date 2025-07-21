import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import json
import pickle
import os
from datetime import datetime

# [Previous clean_text, load_and_preprocess, prepare_sequences, build_bilstm_model functions remain the same]

def save_model_artifacts(model, tokenizer, X_sample, save_dir='models/saved_models/bilstm_model'):
    """Saves all model components with proper organization"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Save full Keras model (architecture + weights + optimizer state)
    model_path = os.path.join(save_dir, 'bilstm_cyberbullying_model.keras')
    save_model(model, model_path)
    
    # 2. Save tokenizer with full configuration
    tokenizer_path = os.path.join(save_dir, 'tokenizer.pkl')
    with open(tokenizer_path, 'wb') as handle:
        pickle.dump({
            'tokenizer': tokenizer,
            'config': {
                'num_words': tokenizer.num_words,
                'oov_token': tokenizer.oov_token,
                'document_count': tokenizer.document_count
            }
        }, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # 3. Save model configuration
    config = {
        'model_type': 'BiLSTM',
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'input_shape': model.input_shape,
        'vocab_size': tokenizer.num_words,
        'max_sequence_length': X_sample.shape[1],
        'layers': [layer.get_config() for layer in model.layers],
        'training_metrics': {
            'loss': model.history.history['loss'][-1],
            'accuracy': model.history.history['accuracy'][-1],
            'val_loss': model.history.history['val_loss'][-1],
            'val_accuracy': model.history.history['val_accuracy'][-1]
        }
    }
    
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # 4. Save sample predictions for verification
    sample_texts = ["good morning", "you're stupid", "I hate you 😠"]
    sample_clean = [clean_text(t) for t in sample_texts]
    sample_seq = tokenizer.texts_to_sequences(sample_clean)
    sample_pad = pad_sequences(sample_seq, maxlen=X_sample.shape[1], padding='post')
    sample_preds = model.predict(sample_pad)
    
    samples_path = os.path.join(save_dir, 'samples.json')
    with open(samples_path, 'w') as f:
        json.dump({
            'texts': sample_texts,
            'predictions': sample_preds.tolist(),
            'clean_versions': sample_clean
        }, f, indent=4)
    
    print(f"✅ Model artifacts saved to {save_dir}:")
    print(f"   - Model: bilstm_cyberbullying_model.keras")
    print(f"   - Tokenizer: tokenizer.pkl")
    print(f"   - Config: config.json")
    print(f"   - Samples: samples.json")

def main():
    # [Previous data loading and preprocessing remains the same until after model training]
    
    # After model.fit()
    
    # Define save directory
    save_dir = os.path.join('models', 'saved_models', 'bilstm_model')
    
    # Enhanced saving
    save_model_artifacts(model, tokenizer, X_train, save_dir)
    
    # Verification loading test
    print("\nRunning verification load test...")
    try:
        loaded_model = tf.keras.models.load_model(os.path.join(save_dir, 'bilstm_cyberbullying_model.keras'))
        with open(os.path.join(save_dir, 'tokenizer.pkl'), 'rb') as f:
            loaded_tokenizer = pickle.load(f)['tokenizer']
        
        test_text = "This is a test 😊"
        test_seq = loaded_tokenizer.texts_to_sequences([clean_text(test_text)])
        test_pad = pad_sequences(test_seq, maxlen=X_train.shape[1])
        pred = loaded_model.predict(test_pad)[0][0]
        
        print(f"✅ Verification passed! Prediction for '{test_text}': {'TOXIC' if pred > 0.5 else 'NOT TOXIC'}")
    except Exception as e:
        print(f"❌ Verification failed: {str(e)}")

if __name__ == "__main__":
    main()