import sys
from pathlib import Path
import pandas as pd
import dill
from scipy import sparse

# Add project root to sys.path if needed
sys.path.insert(0, str(Path('.').resolve()))

# Load training texts from CSV
df = pd.read_csv("data/train.csv")
train_texts = df['text'].dropna().astype(str).tolist()

# Load your saved Logistic Regression pipeline
pipeline_path = Path("models/saved_models/lr_pipeline.pkl")
with open(pipeline_path, 'rb') as f:
    lr_pipeline = dill.load(f)

# Vectorize the training texts
vectorizer = lr_pipeline.named_steps['tfidf']
X_train_vect = vectorizer.transform(train_texts)

# Optionally take a sample if too large
sample_size = 1000
if X_train_vect.shape[0] > sample_size:
    X_train_sample = X_train_vect[:sample_size]
else:
    X_train_sample = X_train_vect

# Save the vectorized sample for SHAP
save_path = Path("models/saved_models/x_train_sample.npz")
sparse.save_npz(save_path, X_train_sample)

print(f"✅ Saved vectorized training sample for SHAP at {save_path}")