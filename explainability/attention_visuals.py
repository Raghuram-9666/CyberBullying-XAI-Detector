import matplotlib.pyplot as plt
import numpy as np
import torch
from pipeline import shared_state

def visualize_xlmr_attention(text, layer=0, head=0):
    """
    Enhanced attention visualization for XLM-RoBERTa using shared_state
    
    Args:
        text: Input text to visualize
        layer: Which layer to visualize (default: 0)
        head: Which attention head to visualize (default: 0)
    
    Returns:
        matplotlib Figure object
    """
    try:
        # Get model and tokenizer from shared state
        model = shared_state.XLM_MODEL
        tokenizer = shared_state.XLM_TOKENIZER

        # Tokenize with truncation
        inputs = tokenizer(text, 
                         return_tensors="pt", 
                         truncation=True, 
                         max_length=512)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get attention weights
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            attentions = torch.stack(outputs.attentions)  # [layers, batch, heads, seq, seq]
        
        # Select specific layer and head
        attention_weights = attentions[layer, 0, head].cpu().numpy()
        
        # Process tokens
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        tokens = [t.replace('▁', ' ') for t in tokens]  # Clean subword tokens
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(attention_weights, cmap="viridis", aspect="auto")
        
        # Formatting
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(tokens, fontsize=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.7)
        cbar.set_label("Attention Weight", rotation=270, labelpad=15)
        
        plt.title(f"Attention Heatmap (Layer {layer+1}, Head {head+1})", pad=20)
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Attention visualization error: {str(e)}")
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.text(0.5, 0.5, "Attention visualization failed", ha='center', va='center')
        return fig



def visualize_session_attention(messages, attention_scores, top_k=3):
    try:
        if not messages or len(messages) != len(attention_scores):
            raise ValueError("Messages and attention_scores must be the same length.")

        # Convert attention_scores to list if it's a numpy array
        if hasattr(attention_scores, 'tolist'):
            attention_scores = attention_scores.tolist()

        print(f"🧪 [AttentionVisuals] Messages: {messages}")
        print(f"🧪 [AttentionVisuals] Raw scores: {attention_scores}")

        fig, ax = plt.subplots(figsize=(12, 4))
        x_pos = np.arange(len(messages))

        max_score = max(attention_scores) if attention_scores else 1
        norm_scores = np.array(attention_scores) / max_score if max_score != 0 else np.zeros_like(attention_scores)

        colors = plt.cm.Reds(norm_scores * 0.7 + 0.3)

        bars = ax.bar(x_pos, attention_scores, color=colors, edgecolor='black')

        # Highlight and annotate top messages
        if attention_scores:
            sorted_indices = np.argsort(attention_scores)[-top_k:]
            for i in sorted_indices:
                ax.text(i, attention_scores[i] + 0.02, 
                        f"{attention_scores[i]:.2f}", 
                        ha='center', 
                        fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"Msg {i+1}" for i in range(len(messages))], rotation=45)
        ax.set_ylabel("Attention Score", fontsize=12)
        ax.set_title("Session Attention Distribution", pad=20, fontsize=14)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

        plt.tight_layout()
        print("✅ Attention plot created successfully.")
        return fig

    except Exception as e:
        print(f"❌ Attention visualization error: {e}")
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.text(0.5, 0.5, "Session visualization failed", ha='center', va='center')
        return fig
