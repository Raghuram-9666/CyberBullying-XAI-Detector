import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Input, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import re
import emoji
import os
import pickle
from google.colab import drive
import warnings
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')

# Mount Google Drive
drive.mount('/content/drive')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
# Create directories for saving results (Updated to match Code 1)
def create_directories():
    base_dir = '/content/drive/MyDrive/cyber_bullying'
    model_dir = os.path.join(base_dir, 'bilstm-toxic-classifier')
    metrics_dir = os.path.join(base_dir, 'bilstm_training_metrics')
    graphs_dir = os.path.join(base_dir, 'bilstm_training_graphs')

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

    return model_dir, metrics_dir, graphs_dir

# Text cleaning function (Same as Code 2)
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = emoji.demojize(text, delimiters=("", ""))
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    text = re.sub(r'l0ser|los3r', 'loser', text)
    return text.strip()

# Load and preprocess data (Same as Code 2)
def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    df['clean_text'] = df['text'].apply(clean_text)
    df['label'] = df['toxic']
    df = df[df['clean_text'].str.len() > 0]

    print(f"Label distribution:")
    print(df['label'].value_counts())

    return df

def create_message_sequences(df, sequence_length=5):
    sequences = []
    labels = []

    df = df.copy()
    df = df[df['clean_text'].str.len() > 0]

    half_len = sequence_length // 2
    toxic_sequences = []
    non_toxic_sequences = []

    grouped = df.groupby('dialogue_id')

    for _, group in grouped:
        group_sorted = group.sort_values('turn_id').reset_index(drop=True)
        messages = group_sorted['clean_text'].tolist()
        labels_list = group_sorted['label'].tolist()

        for i in range(len(messages)):
            start = max(0, i - half_len)
            end = start + sequence_length
            if end <= len(messages):
                seq = messages[start:end]
                if labels_list[i] == 1:
                    toxic_sequences.append(seq)
                elif labels_list[i] == 0:
                    non_toxic_sequences.append(seq)

    # Balance classes
    min_count = min(len(toxic_sequences), len(non_toxic_sequences))
    toxic_sequences = toxic_sequences[:min_count]
    non_toxic_sequences = non_toxic_sequences[:min_count]

    sequences = toxic_sequences + non_toxic_sequences
    labels = [1] * min_count + [0] * min_count

    combined = list(zip(sequences, labels))
    np.random.shuffle(combined)
    sequences, labels = zip(*combined)

    return list(sequences), list(labels)

# Prepare sequences (Same as Code 2)
def prepare_sequences(message_sequences, max_len=50, num_words=10000):
    all_messages = []
    for seq in message_sequences:
        all_messages.extend(seq)

    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(all_messages)

    processed_sequences = []
    for seq in message_sequences:
        tokenized_seq = []
        for message in seq:
            tokens = tokenizer.texts_to_sequences([message])[0]
            padded_tokens = pad_sequences([tokens], maxlen=max_len, padding='post', truncating='post')[0]
            tokenized_seq.append(padded_tokens)
        processed_sequences.append(tokenized_seq)

    return np.array(processed_sequences), tokenizer

# Build hierarchical BiLSTM model (Same as Code 2)
def build_hierarchical_bilstm(vocab_size, embedding_dim=100, message_max_len=50, sequence_length=3):
    input_layer = Input(shape=(sequence_length, message_max_len))

    message_encoder = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=message_max_len),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.3)
    ])

    encoded_messages = TimeDistributed(message_encoder)(input_layer)

    sequence_lstm = Bidirectional(LSTM(64, return_sequences=False))(encoded_messages)
    sequence_dropout = Dropout(0.3)(sequence_lstm)

    dense1 = Dense(32, activation='relu')(sequence_dropout)
    dropout2 = Dropout(0.3)(dense1)
    output = Dense(1, activation='sigmoid')(dropout2)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# Custom callback (Same as Code 2)
class MetricsHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        self.train_acc = []
        self.val_acc = []
        self.train_loss = []
        self.val_loss = []
        self.epochs = []

    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch + 1)
        self.train_acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

# Visualization functions (Same as Code 2)
def plot_training_history(history, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(history.epochs, history.train_loss, 'b-', label='Training Loss')
    axes[0, 0].plot(history.epochs, history.val_loss, 'r-', label='Validation Loss')
    axes[0, 0].set_title('Training vs Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(history.epochs, history.train_acc, 'b-', label='Training Accuracy')
    axes[0, 1].plot(history.epochs, history.val_acc, 'r-', label='Validation Accuracy')
    axes[0, 1].set_title('Training vs Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].remove()
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

def save_classification_report(y_true, y_pred, save_dir, class_names=['Non-Toxic', 'Toxic']):
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_str = classification_report(y_true, y_pred, target_names=class_names)

    df_report = pd.DataFrame(report_dict).transpose()
    df_report = df_report.round(3)

    df_report.to_csv(os.path.join(save_dir, 'classification_report.csv'))

    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write("="*60 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n")
        f.write(report_str)
        f.write("\n" + "="*60 + "\n")
        f.write("\nDetailed DataFrame Format:\n")
        f.write("-" * 40 + "\n")
        f.write(str(df_report))
        f.write("\n" + "="*60 + "\n")

    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(report_str)
    print("="*60)

def save_training_metrics(history, save_dir):
    metrics_df = pd.DataFrame({
        'epoch': history.epochs,
        'train_loss': history.train_loss,
        'val_loss': history.val_loss,
        'train_accuracy': history.train_acc,
        'val_accuracy': history.val_acc
    })

    metrics_df.to_csv(os.path.join(save_dir, 'training_metrics.csv'), index=False)

    best_epoch = np.argmax(history.val_acc) + 1
    summary_stats = {
        'best_epoch': int(best_epoch),
        'best_val_accuracy': float(max(history.val_acc)),
        'final_train_loss': float(history.train_loss[-1]),
        'final_val_loss': float(history.val_loss[-1]),
        'final_train_accuracy': float(history.train_acc[-1]),
        'final_val_accuracy': float(history.val_acc[-1])
    }

    with open(os.path.join(save_dir, 'training_summary.txt'), 'w') as f:
        f.write("TRAINING SUMMARY\n")
        f.write("="*30 + "\n")
        for key, value in summary_stats.items():
            f.write(f"{key}: {value}\n")

    print(f"Training metrics saved to {save_dir}")
    return summary_stats

# Main function (Same as Code 2)
def main():
    model_dir, metrics_dir, graphs_dir = create_directories()

    csv_path = '/content/drive/MyDrive/cyber_bullying/data/train.csv'
    df = load_and_preprocess(csv_path)

    sequence_length = 3
    message_sequences, sequence_labels = create_message_sequences(df, sequence_length)

    print(f"Created {len(message_sequences)} message sequences")
    print(f"Sequence label distribution: {np.bincount(sequence_labels)}")

    max_len = 50
    vocab_size = 10000
    embedding_dim = 100

    X_sequences, tokenizer = prepare_sequences(message_sequences, max_len=max_len, num_words=vocab_size)
    y_sequences = np.array(sequence_labels)

    print(f"Sequence input shape: {X_sequences.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_sequences, y_sequences, test_size=0.2, random_state=42, stratify=y_sequences
    )

    print(f"Training sequences: {len(X_train)}")
    print(f"Test sequences: {len(X_test)}")

    model = build_hierarchical_bilstm(vocab_size, embedding_dim, max_len, sequence_length)
    print("Model Architecture:")
    model.summary()

    metrics_history = MetricsHistory()
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)

    print("Starting training...")
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    history = model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[metrics_history, early_stopping, reduce_lr],
        class_weight=class_weight_dict,
        verbose=1
    )


    model_path = os.path.join(model_dir, 'bilstm_sequence_model.keras')
    tokenizer_path = os.path.join(model_dir, 'tokenizer.pkl')

    model.save(model_path)
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)

    print(f"Model saved to {model_path}")
    print(f"Tokenizer saved to {tokenizer_path}")

    print("\nGenerating predictions...")
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)

    print("\nSaving training metrics...")
    summary_stats = save_training_metrics(metrics_history, metrics_dir)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    final_results = pd.DataFrame({
        'true_label': y_test,
        'predicted_label': y_pred,
        'prediction_probability': y_pred_prob.flatten()
    })
    final_results.to_csv(os.path.join(metrics_dir, 'final_predictions.csv'), index=False)

    print("\nGenerating and saving visualizations...")
    plot_training_history(metrics_history, graphs_dir)
    plot_confusion_matrix(y_test, y_pred, graphs_dir)

    # ✅ Safely handle classification report
    if len(np.unique(y_test)) > 1 and len(np.unique(y_pred)) > 1:
      save_classification_report(y_test, y_pred, metrics_dir)
    else:
      print("⚠️ Skipped classification report — only one class present in predictions.")

    plot_roc_curve(y_test, y_pred_prob.flatten(), graphs_dir)


    print("\nTraining completed successfully!")
    print(f"Final test accuracy: {accuracy:.4f}")
    print(f"Final test precision: {precision:.4f}")
    print(f"Final test recall: {recall:.4f}")
    print(f"Final test F1: {f1:.4f}")
    print(f"\nFiles saved in:")
    print(f"  - Model: {model_dir}")
    print(f"  - Metrics: {metrics_dir}")
    print(f"  - Graphs: {graphs_dir}")

    return model, tokenizer, summary_stats

if __name__ == "__main__":
    model, tokenizer, stats = main()