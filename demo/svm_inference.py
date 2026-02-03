import dill
import emoji
import re
from pathlib import Path
import sys
import os
import json
from typing import Union, List, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from explainability.lime_explainer import explain_svm
from pipeline import shared_state

class ToxicityClassifier:
    def __init__(self, model_path: str = 'models/saved_models/svm_pipeline.pkl'):
        self.model_path = model_path
        self.pipeline = None
        self.is_loaded = False
        self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        with open(self.model_path, 'rb') as f:
            self.pipeline = dill.load(f)
        self.is_loaded = True
        shared_state.SVM_MODEL = self.pipeline  # Set for LIME
        print(f"✅ SVM model loaded successfully from {self.model_path}")

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
                result = {
                    'text': t,
                    'is_toxic': bool(prediction),
                    'confidence': float(max(probabilities)),
                    'probabilities': {
                        'not_toxic': float(probabilities[0]),
                        'toxic': float(probabilities[1])
                    },
                    'emoji_info': emoji_info,
                    'error': None
                }
            except Exception as e:
                result = {
                    'text': t,
                    'is_toxic': False,
                    'confidence': 0.0,
                    'probabilities': None,
                    'emoji_info': None,
                    'error': str(e)
                }
            results.append(result)
        return results[0] if single_input else results

    def predict_with_interpretation(self, text: str) -> Dict:
        result = self.predict(text)
        if result.get('error'):
            return result

        if result['is_toxic']:
            if result['confidence'] > 0.9:
                interpretation = "High confidence toxicity detected"
            elif result['confidence'] > 0.7:
                interpretation = "Moderate confidence toxicity detected"
            else:
                interpretation = "Low confidence toxicity detected"
            if result['emoji_info']['negative_emoji_count'] > 0:
                interpretation += f" (negative emojis detected: {result['emoji_info']['negative_emoji_count']})"
        else:
            if result['confidence'] > 0.9:
                interpretation = "High confidence non-toxic"
            elif result['confidence'] > 0.7:
                interpretation = "Moderate confidence non-toxic"
            else:
                interpretation = "Low confidence non-toxic"
            if result['emoji_info']['positive_emoji_count'] > 0:
                interpretation += f" (positive emojis detected: {result['emoji_info']['positive_emoji_count']})"
        result['interpretation'] = interpretation
        return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Toxicity Detection Inference with SVM + LIME')
    parser.add_argument('--text', type=str, help='Single text to analyze')
    parser.add_argument('--file', type=str, help='File containing texts to analyze (one per line)')
    parser.add_argument('--model', type=str, default='models/saved_models/svm_pipeline.pkl',
                        help='Path to SVM model file')
    parser.add_argument('--output', type=str, help='Output file for results (JSON)')
    parser.add_argument('--detailed', action='store_true', help='Show detailed analysis including LIME explanation')
    args = parser.parse_args()

    try:
        detector = ToxicityClassifier(args.model)
    except Exception as e:
        print(f"❌ Failed to initialize detector: {str(e)}")
        return

    results = []

    if args.text:
        result = detector.predict_with_interpretation(args.text) if args.detailed else detector.predict(args.text)
        results.append(result)

        print(f"'{result['text']}' => {'🔴 TOXIC' if result['is_toxic'] else '🟢 NOT TOXIC'} (confidence: {result['confidence']:.3f})")
        if args.detailed:
            print(f"  Interpretation: {result.get('interpretation', 'N/A')}")
            print(f"  Emojis: {result['emoji_info']['emoji_count']}")
            if result['emoji_info']['emoji_list']:
                print(f"  Emoji list: {', '.join(result['emoji_info']['emoji_list'])}")

        if args.detailed and result['is_toxic'] is not None:
            try:
                os.makedirs("results/lime", exist_ok=True)
                fig, exp_list = explain_svm(result["text"])
                filename = re.sub(r'[^\w]', '_', result["text"][:30]) + ".png"
                lime_path = os.path.join("results/lime", filename)
                fig.savefig(lime_path)
                print(f"📊 LIME explanation saved to: {lime_path}")
                plt.close(fig)
            except Exception as e:
                print(f"⚠️ Could not generate LIME explanation: {str(e)}")

    elif args.file:
        if not os.path.exists(args.file):
            print(f"❌ File not found: {args.file}")
            return
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"📄 Processing {len(texts)} texts from {args.file}")
        results = [detector.predict_with_interpretation(text) if args.detailed else detector.predict(text) for text in texts]

    else:
        print("🔍 Interactive Toxicity Detection (SVM + LIME)")
        print("Type 'quit' to exit")
        while True:
            text = input("\nEnter text to analyze: ").strip()
            if text.lower() == 'quit':
                break
            if text:
                try:
                    result = detector.predict_with_interpretation(text) if args.detailed else detector.predict(text)
                    print(f"Result: {'🔴 TOXIC' if result['is_toxic'] else '🟢 NOT TOXIC'}")
                    print(f"Confidence: {result['confidence']:.3f}")
                    if args.detailed:
                        print(f"Interpretation: {result.get('interpretation', 'N/A')}")
                        print(f"Emojis detected: {result['emoji_info']['emoji_count']}")
                        if result['emoji_info']['emoji_list']:
                            print(f"Emoji list: {', '.join(result['emoji_info']['emoji_list'])}")
                    if args.detailed and result['is_toxic'] is not None:
                        try:
                            os.makedirs("results/lime", exist_ok=True)
                            fig, exp_list = explain_svm(result["text"])
                            filename = re.sub(r'[^\w]', '_', result["text"][:30]) + ".png"
                            lime_path = os.path.join("results/lime", filename)
                            fig.savefig(lime_path)
                            print(f"📊 LIME explanation saved to: {lime_path}")
                            plt.close(fig)
                        except Exception as e:
                            print(f"⚠️ Could not generate LIME explanation: {str(e)}")
                except Exception as e:
                    print(f"Error during prediction: {str(e)}")

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Results saved to {args.output}")


if __name__ == '__main__':
    import matplotlib.pyplot as plt  # Needed for closing fig after saving
    main()