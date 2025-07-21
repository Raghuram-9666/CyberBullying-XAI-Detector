import torch
import re
import emoji
import os
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

def setup_device():
    """Check and setup the available device"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using {device} device")
    return device

def load_model_and_tokenizer(model_path):
    """Load the pre-trained model and tokenizer from local directory"""
    try:
        abs_path = os.path.abspath(model_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Model directory not found: {abs_path}")
        
        print(f"📂 Loading model from: {abs_path}")
        
        # No need to manually check for pytorch_model.bin—transformers handles this
        tokenizer = XLMRobertaTokenizer.from_pretrained(abs_path)
        model = XLMRobertaForSequenceClassification.from_pretrained(abs_path)
        
        print("✅ Model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None, None

def clean_text(text):
    """Clean and preprocess the input text"""
    text = emoji.demojize(text, delimiters=("", ""))
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    text = re.sub(r"l0ser|los3r", "loser", text)
    return text.strip()

def predict_toxicity(model, tokenizer, device, text):
    """Predict if the text is toxic or not"""
    cleaned_text = clean_text(text)
    
    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        toxic_prob = probabilities[0][1].item()
    
    return {
        "text": text,
        "cleaned": cleaned_text,
        "is_toxic": toxic_prob > 0.5,
        "toxicity_score": round(toxic_prob, 4)
    }

def main():
    # ✅ Relative path to model directory (adjust according to where you save this script)
    MODEL_PATH = os.path.join("..", "cyberbullying_detection_project", "models", "saved_models", "toxic-comment-classifier")
    
    device = setup_device()
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    
    if model is None:
        print("\n⚠️ Please check:")
        print(f"1. Does this folder exist? --> {os.path.abspath(MODEL_PATH)}")
        print("2. Does it contain: config.json, model.safetensors, tokenizer files")
        return
    
    model.to(device)
    model.eval()
    
    print("\n✅ Toxicity classifier ready. Type your message or 'exit' to quit.")
    
    while True:
        user_input = input("\nEnter message: ").strip()
        
        if user_input.lower() in ['exit', 'quit']:
            print("👋 Exiting. Goodbye!")
            break
            
        if not user_input:
            print("⚠️ Please enter some text")
            continue
            
        result = predict_toxicity(model, tokenizer, device, user_input)
        
        print("\n📝 Results:")
        print(f"Original: {result['text']}")
        print(f"Cleaned: {result['cleaned']}")
        print(f"Prediction: {'🚨 TOXIC' if result['is_toxic'] else '✅ CLEAN'}")
        print(f"Confidence: {result['toxicity_score']:.1%}")

if __name__ == "__main__":
    main()