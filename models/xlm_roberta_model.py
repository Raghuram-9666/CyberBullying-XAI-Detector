import pandas as pd
import numpy as np
import re
import emoji
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from transformers import (
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
    TrainerCallback
)
from datasets import Dataset
import torch
from google.colab import drive
import warnings
warnings.filterwarnings('ignore')

# Create directories for saving results
def create_directories():
    model_dir = './toxic-comment-classifier'
    metrics_dir = './training_metrics'
    graphs_dir = './training_graphs'

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

    return model_dir, metrics_dir, graphs_dir

# Set seed for reproducibility
set_seed(42)

# Mount Google Drive (for Colab)
drive.mount('/content/drive')

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Custom callback to collect training metrics
class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []
        self.epochs = []

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs:
            # Training loss
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])

            # Validation metrics
            if 'eval_loss' in logs:
                self.val_losses.append(logs['eval_loss'])
                self.epochs.append(state.epoch)

            if 'eval_accuracy' in logs:
                self.val_accuracies.append(logs['eval_accuracy'])

            if 'eval_f1' in logs:
                self.val_f1_scores.append(logs['eval_f1'])

# Text cleaning
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = emoji.demojize(text, delimiters=("", ""))
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    text = re.sub(r"l0ser|los3r", "loser", text)
    return text.strip()

# Load and preprocess data
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Clean text
    df['clean_text'] = df['text'].apply(clean_text)

    # Use the toxic column directly (assuming it's binary 0/1)
    df['label'] = df['toxic']

    # Remove empty texts
    df = df[df['clean_text'].str.len() > 0]

    print(f"Label distribution:")
    print(df['label'].value_counts())

    return df[['clean_text', 'label']]

# Tokenization
def tokenize_function(examples, tokenizer):
    return tokenizer(examples['clean_text'], truncation=True, padding='max_length', max_length=128)

# Compute metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    return {
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1': report['weighted avg']['f1-score'],
        'accuracy': report['accuracy']
    }

# Visualization functions
def plot_training_history(callback, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Training vs Validation Loss
    axes[0, 0].plot(callback.epochs, callback.train_losses[:len(callback.epochs)], 'b-', label='Training Loss')
    axes[0, 0].plot(callback.epochs, callback.val_losses, 'r-', label='Validation Loss')
    axes[0, 0].set_title('Training vs Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy per Epoch
    axes[0, 1].plot(callback.epochs, callback.val_accuracies, 'g-', label='Validation Accuracy')
    axes[0, 1].set_title('Validation Accuracy per Epoch')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # F1 Score per Epoch
    axes[1, 0].plot(callback.epochs, callback.val_f1_scores, 'm-', label='Validation F1')
    axes[1, 0].set_title('Validation F1 Score per Epoch')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Remove empty subplot
    axes[1, 1].remove()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, save_dir, class_names=['Non-Toxic', 'Toxic']):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curve(y_true, y_proba, save_dir):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.show()

def save_classification_report_table(y_true, y_pred, save_dir, class_names=['Non-Toxic', 'Toxic']):
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # Create DataFrame for better visualization
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report.round(3)

    # Save to CSV
    df_report.to_csv(os.path.join(save_dir, 'classification_report.csv'))

    # Save as formatted text
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write("="*50 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*50 + "\n")
        f.write(str(df_report))
        f.write("\n" + "="*50 + "\n")

    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(df_report)
    print("="*50)

def save_training_metrics(callback, save_dir):
    # Create metrics DataFrame
    metrics_df = pd.DataFrame({
        'epoch': callback.epochs,
        'train_loss': callback.train_losses[:len(callback.epochs)],
        'val_loss': callback.val_losses,
        'val_accuracy': callback.val_accuracies,
        'val_f1_score': callback.val_f1_scores
    })

    # Save to CSV
    metrics_df.to_csv(os.path.join(save_dir, 'training_metrics.csv'), index=False)

    # Save summary statistics
    summary_stats = {
        'best_epoch': int(callback.epochs[np.argmax(callback.val_f1_scores)]),
        'best_val_f1': float(max(callback.val_f1_scores)),
        'best_val_accuracy': float(max(callback.val_accuracies)),
        'final_train_loss': float(callback.train_losses[-1]),
        'final_val_loss': float(callback.val_losses[-1])
    }

    # Save summary to text file
    with open(os.path.join(save_dir, 'training_summary.txt'), 'w') as f:
        f.write("TRAINING SUMMARY\n")
        f.write("="*30 + "\n")
        for key, value in summary_stats.items():
            f.write(f"{key}: {value}\n")

    print(f"Training metrics saved to {save_dir}")
    return summary_stats

# Main function
def main():
    # Create directories
    model_dir, metrics_dir, graphs_dir = create_directories()

    # CSV file location in Colab
    csv_path = '/content/drive/MyDrive/cyber_bullying/data/train.csv'

    # Load data
    df = load_data(csv_path)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")

    # Create datasets
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    # Load tokenizer and model
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=2)
    model.to(device)

    # Tokenize datasets
    train_tokenized = train_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_tokenized = val_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Initialize callback for metrics collection
    metrics_callback = MetricsCallback()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(model_dir, 'results'),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,  # Adjusted for Colab
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=os.path.join(model_dir, 'logs'),
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        fp16=True,  # Enable mixed precision for GPU
        dataloader_num_workers=2,
        warmup_steps=500,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        compute_metrics=compute_metrics,
        callbacks=[metrics_callback],
    )

    print("Starting training...")
    trainer.train()

    # Save model and tokenizer
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Model saved successfully to {model_dir}!")

    # Generate predictions for evaluation
    print("\nGenerating predictions...")
    predictions = trainer.predict(val_tokenized)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = val_df['label'].values
    y_proba = predictions.predictions[:, 1]  # Probabilities for positive class

    # Save training metrics
    print("\nSaving training metrics...")
    summary_stats = save_training_metrics(metrics_callback, metrics_dir)

    # Plot and save visualizations
    print("\nGenerating and saving visualizations...")

    # 1. Training History
    plot_training_history(metrics_callback, graphs_dir)

    # 2. Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, graphs_dir)

    # 3. Classification Report
    save_classification_report_table(y_true, y_pred, metrics_dir)

    # 4. ROC Curve
    plot_roc_curve(y_true, y_proba, graphs_dir)

    # Save final predictions
    final_results = pd.DataFrame({
        'true_label': y_true,
        'predicted_label': y_pred,
        'prediction_probability': y_proba
    })
    final_results.to_csv(os.path.join(metrics_dir, 'final_predictions.csv'), index=False)

    print("\nTraining completed successfully!")
    print(f"Final validation accuracy: {predictions.metrics['test_accuracy']:.4f}")
    print(f"Final validation F1: {predictions.metrics['test_f1']:.4f}")
    print(f"\nFiles saved in:")
    print(f"  - Model: {model_dir}")
    print(f"  - Metrics: {metrics_dir}")
    print(f"  - Graphs: {graphs_dir}")

    return summary_stats

if __name__ == "__main__":
    main()