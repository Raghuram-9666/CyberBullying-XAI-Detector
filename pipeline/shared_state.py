# Add these imports at the top
import torch
import tensorflow as tf
from transformers import AutoModelForSequenceClassification
from sklearn.pipeline import Pipeline

# Shared resources
MODELS_LOADED = False

# XLM-RoBERTa
XLM_TOKENIZER = None
XLM_MODEL = None

# BiLSTM
BILSTM_TOKENIZER = None
BILSTM_MODEL = None

# Classical Models (updated types)
SVM_MODEL = None          # Will hold SVMWrapper
SVM_PIPELINE = None       # New: Stores raw pipeline
LR_MODEL = None           # Will hold LRWrapper 
LR_PIPELINE = None        # New: Stores raw pipeline

# Vectorizer (now part of pipelines)
VECTORIZER = None         # Will reference SVM_PIPELINE['tfidf']

# Explainability data
X_TRAIN_SAMPLE = None     # Sample data for SHAP