import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dill
import pandas as pd
import emoji
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.persistent_vectorizer import PersistentVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score

def custom_tokenizer(text):
    """Tokenize text while preserving emojis"""
    text = emoji.demojize(text, delimiters=(" ", " "))  # 😠 → ":angry_face:"
    return re.findall(r"\b\w+\b|:[a-z_]+:", text.lower())

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['NOT TOXIC', 'TOXIC'], 
                yticklabels=['NOT TOXIC', 'TOXIC'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to: {save_path}")
    
    plt.show()
    
    tn, fp, fn, tp = cm.ravel()
    metrics_text = f"""
{title} Metrics:
True Negatives: {tn}
False Positives: {fp}
False Negatives: {fn}
True Positives: {tp}
Precision: {tp/(tp+fp):.3f}
Recall: {tp/(tp+fn):.3f}
Specificity: {tn/(tn+fp):.3f}
Accuracy: {(tp+tn)/(tp+tn+fp+fn):.3f}
    """
    print(metrics_text)
    
    if save_path:
        metrics_file = save_path.replace('.png', '_metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write(metrics_text)

def plot_roc_curve(y_true, y_proba, title="ROC Curve", save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 ROC curve saved to: {save_path}")
    
    plt.show()
    
    auc_text = f"\n{title} AUC Score: {roc_auc:.3f}"
    print(auc_text)
    
    if save_path:
        auc_file = save_path.replace('.png', '_auc.txt')
        with open(auc_file, 'w') as f:
            f.write(auc_text)

def train_svm_model(train_df, val_df, text_col='text', label_col='toxic',
                    model_path='models/saved_models/svm_pipeline.pkl'):
    print(">>> SVM training started")

    # Base LinearSVC (no predict_proba), so use CalibratedClassifierCV for probabilities
    svm = LinearSVC(class_weight='balanced', max_iter=5000)

    # Wrap SVM with calibration to get probabilities
    calibrated_svm = CalibratedClassifierCV(svm, cv=5)

    pipeline = Pipeline([
        ('tfidf', PersistentVectorizer(
            tokenizer=custom_tokenizer,
            ngram_range=(1, 2),
            max_features=5000,
            min_df=5
        )),
        ('svm', calibrated_svm)
    ])

    print("Training model...")
    pipeline.fit(train_df[text_col], train_df[label_col])

    val_preds = pipeline.predict(val_df[text_col])
    val_proba = pipeline.predict_proba(val_df[text_col])[:, 1]

    print("\nValidation Performance:")
    print(classification_report(val_df[label_col], val_preds))

    # Confusion matrix plot
    cm_save_path = "results/plots/svm_model_confusion_matrix.png"
    plot_confusion_matrix(val_df[label_col], val_preds, 
                         "SVM Confusion Matrix", cm_save_path)

    # ROC curve plot
    roc_save_path = "results/plots/svm_model_roc_curve.png"
    plot_roc_curve(val_df[label_col], val_proba, 
                  "SVM ROC Curve", roc_save_path)

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        dill.dump(pipeline, f)
    print(f"\n✅ Model saved to {model_path}")

    # Test emojis and save results
    test_samples = ["Hello 😊", "I hate you 😠", "You're stupid 🤬"]
    results_file = "results/svm_model_test_results.txt"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=== SVM TEST RESULTS ===\n\n")
        f.write("Test Results:\n")
        f.write("-" * 40 + "\n")

        print("\nEmoji Test Results:")
        for text in test_samples:
            pred = pipeline.predict([text])[0]
            proba = pipeline.predict_proba([text])[0]
            result_line = f"{text} => {'TOXIC' if pred else 'NOT TOXIC'} (confidence: {max(proba):.3f})"
            print(result_line)
            f.write(f"{result_line}\n")
            f.write("-" * 40 + "\n")

    print(f"\n💾 Test results saved to: {results_file}")

if __name__ == '__main__':
    train = pd.read_csv('data/train.csv')
    val = pd.read_csv('data/val.csv')
    
    print("Dataset Info:")
    print(f"Train shape: {train.shape}")
    print(f"Val shape: {val.shape}")
    print(f"Columns: {train.columns.tolist()}")
    print(f"Toxic distribution in train: {train['toxic'].value_counts()}")

    train_svm_model(train, val)

    samples = ["You're so stupid! 😠", "That's great! 😊", "I'll kill you 🔪", "You're trash 🤮🤢", "🖕", "Go die 💀🖕"]
    print("\nAdditional Test Results:")
    with open('models/saved_models/svm_pipeline.pkl', 'rb') as f:
        pipeline = dill.load(f)

    for text in samples:
        pred = pipeline.predict([text])[0]
        proba = pipeline.predict_proba([text])[0]
        print(f"{text} => {'TOXIC' if pred else 'NOT TOXIC'} (confidence: {max(proba):.3f})")

    print(f"\n📋 Model Summary:")
    print(f"   • Model Type: LinearSVC with TF-IDF + Calibrated probabilities")
    print(f"   • Features: Text + Emoji Analysis")
    print(f"   • Output: Toxicity prediction with confidence")
    print(f"   • Saved to: models/saved_models/svm_pipeline.pkl")

    print(f"\n🎯 Usage:")
    print("   with open('models/saved_models/svm_pipeline.pkl', 'rb') as f:")
    print("       pipeline = dill.load(f)")
    print("   pipeline.predict(['your text here'])")