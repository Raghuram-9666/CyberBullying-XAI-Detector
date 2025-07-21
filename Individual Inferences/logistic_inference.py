import dill
import emoji
import re
from pathlib import Path
import sys
import os
import json
from typing import Union, List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

class ToxicityClassifier:
    def __init__(self, model_path: str = 'models/saved_models/lr_pipeline.pkl'):
        self.model_path = model_path
        self.pipeline = None
        self.is_loaded = False
        self.load_model()

    def load_model(self):
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            with open(self.model_path, 'rb') as f:
                self.pipeline = dill.load(f)
            self.is_loaded = True
            print(f"✅ Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise e

    @staticmethod
    def custom_tokenizer(text: str) -> List[str]:
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        text = emoji.demojize(text, delimiters=(" ", " "))
        return re.findall(r"\b\w+\b|:[a-z_]+:", text.lower())

    @staticmethod
    def extract_emoji_info(text: str) -> Dict[str, Any]:
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        emojis = emoji.distinct_emoji_list(text)
        negative_emojis = {'😠', '😡', '🤬', '😤', '💢', '🖕', '👎', '💀', '☠️', '🤮', '🤢'}
        positive_emojis = {'😊', '😀', '😃', '❤️', '💕', '👍', '🎉', '😍', '🥰', '😘'}
        return {
            'emoji_count': len(emojis),
            'has_emoji': len(emojis) > 0,
            'negative_emoji_count': sum(1 for e in emojis if e in negative_emojis),
            'positive_emoji_count': sum(1 for e in emojis if e in positive_emojis),
            'emoji_list': emojis
        }

    def predict(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        if not self.is_loaded:
            raise ValueError("Model not loaded - call load_model() first")
        single_input = isinstance(text, str)
        texts = [text] if single_input else text
        results = []
        for t in texts:
            try:
                prediction = self.pipeline.predict([t])[0]
                probabilities = self.pipeline.predict_proba([t])[0]
                emoji_info = self.extract_emoji_info(t)
                labels = ['CLEAN', 'TOXIC', 'SARCASTIC']
                result = {
                    'text': t,
                    'predicted_class': int(prediction),
                    'predicted_label': labels[int(prediction)],
                    'confidence': float(max(probabilities)),
                    'probabilities': {labels[i]: float(probabilities[i]) for i in range(len(probabilities))},
                    'emoji_info': emoji_info,
                    'error': None
                }
            except Exception as e:
                result = {
                    'text': t,
                    'predicted_class': -1,
                    'predicted_label': 'ERROR',
                    'confidence': 0.0,
                    'probabilities': None,
                    'emoji_info': None,
                    'error': str(e)
                }
            results.append(result)
        return results[0] if single_input else results

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        return self.predict(texts)

if __name__ == '__main__':
    model = ToxicityClassifier()
    while True:
        user_input = input("\nEnter text (or type 'exit'): ")
        if user_input.lower() == 'exit':
            break
        prediction = model.predict(user_input)
        print(json.dumps(prediction, indent=2, ensure_ascii=False))