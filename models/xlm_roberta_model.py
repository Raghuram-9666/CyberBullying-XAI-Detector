import pandas as pd
import numpy as np
import re
import emoji
import torch
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import (
    XLMRobertaTokenizer, 
    XLMRobertaForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed
)
from datasets import Dataset

# Set seed for reproducibility
set_seed(42)

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"GPU name: {torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'}")

# Enhanced text cleaning
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = emoji.demojize(text, delimiters=("", ""))
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"l0ser|los3r", "loser", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    return text.strip()

# Load and preprocess data
def load_and_prepare_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df['clean_text'] = df['text'].apply(clean_text)
    df['labels'] = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].max(axis=1).astype(int)
    return df[['clean_text', 'labels']]

# Tokenization
def tokenize_data(dataset, tokenizer):
    return dataset.map(
        lambda x: tokenizer(
            x['clean_text'], 
            truncation=True, 
            padding='max_length', 
            max_length=128
        ),
        batched=True,
        remove_columns=['clean_text']  # Remove text column after tokenization
    )

# Enhanced metrics computation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    return {
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1': report['weighted avg']['f1-score'],
        'accuracy': report['accuracy'],
    }

def main():
    # Load data
    df = load_and_prepare_dataset('data/train.csv')
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Convert to Hugging Face dataset format
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    # Load tokenizer and model
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    model = XLMRobertaForSequenceClassification.from_pretrained(
        'xlm-roberta-base', 
        num_labels=2
    ).to(device)

    # Tokenize datasets
    train_tokenized = tokenize_data(train_ds, tokenizer)
    val_tokenized = tokenize_data(val_ds, tokenizer)

    # Enhanced training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16 if device == "cuda" else 8,
        per_device_eval_batch_size=64 if device == "cuda" else 32,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        warmup_ratio=0.1,
        report_to="none",
        save_total_limit=2,
    )

    # Initialize Trainer with early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Train and evaluate
    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")

    # Save model and tokenizer
    save_dir = 'models/saved_models/toxic-comment-classifier/'
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")

if __name__ == "__main__":
    main()
