import time
import torch
import dill 
import pickle
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pipeline import shared_state
from scipy import sparse
shared_state.X_TRAIN_SAMPLE = sparse.load_npz("models/saved_models/x_train_sample.npz")

def load_models():
    if not shared_state.MODELS_LOADED:
        print("⚙️ Loading all models...")
        start = time.time()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            # 1. Load XLM-R
            xlm_path = "models/saved_models/toxic-comment-classifier"
            shared_state.XLM_TOKENIZER = AutoTokenizer.from_pretrained(xlm_path)
            shared_state.XLM_MODEL = AutoModelForSequenceClassification.from_pretrained(xlm_path).to(device)
            shared_state.XLM_MODEL.eval()

            # 2. Load BiLSTM
            bilstm_path = "models/saved_models/bilstm_model"
            shared_state.BILSTM_MODEL = tf.keras.models.load_model(f"{bilstm_path}/bilstm_sequence_model.keras")
            with open(f"{bilstm_path}/tokenizer.pkl", 'rb') as f:
                shared_state.BILSTM_TOKENIZER = pickle.load(f)

            # 3. Load Classical Models
            with open("models/saved_models/svm_pipeline.pkl", 'rb') as f:
                shared_state.SVM_PIPELINE = dill.load(f)
            
            with open("models/saved_models/lr_pipeline.pkl", 'rb') as f:
                shared_state.LR_PIPELINE = dill.load(f)
            
            # Initialize wrappers
            SVMWrapper, LRWrapper = get_wrappers()
            shared_state.SVM_MODEL = SVMWrapper(shared_state.SVM_PIPELINE)
            shared_state.LR_MODEL = LRWrapper(shared_state.LR_PIPELINE)
            
            # Set vectorizer reference
            shared_state.VECTORIZER = shared_state.LR_PIPELINE.named_steps['tfidf']
            from scipy import sparse
            shared_state.X_TRAIN_SAMPLE = sparse.load_npz("models/saved_models/x_train_sample.npz")

            shared_state.MODELS_LOADED = True
            print(f"✅ All models loaded in {time.time()-start:.2f}s on {device}")
            
        except Exception as e:
            print(f"❌ Loading failed: {str(e)}")
            raise

def get_wrappers():
    """Lazy import to break circular dependencies"""
    from pipeline.message_pipeline import SVMWrapper, LRWrapper
    return SVMWrapper, LRWrapper

def check_shared_state(verbose=True):
    """Verify all required models are loaded in shared_state"""
    required_attrs = {
        'XLM': ['XLM_TOKENIZER', 'XLM_MODEL'],
        'BiLSTM': ['BILSTM_TOKENIZER', 'BILSTM_MODEL'],
        'SVM': ['SVM_PIPELINE', 'SVM_MODEL'],
        'LR': ['LR_PIPELINE', 'LR_MODEL'],
        'Vectorizer': ['VECTORIZER']
    }
    
    all_valid = True
    for group, attrs in required_attrs.items():
        missing = [attr for attr in attrs if not hasattr(shared_state, attr)]
        if missing:
            print(f"❌ {group} missing: {', '.join(missing)}")
            all_valid = False
        elif verbose:
            print(f"✅ {group} fully loaded")
    
    if all_valid:
        print("🔥 All models verified successfully!")
    return all_valid