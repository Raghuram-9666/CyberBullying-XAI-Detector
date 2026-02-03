import time
import dill
from pipeline import shared_state
from pipeline.message_pipeline import process_message
from pipeline.session_pipeline import process_session
from utils.load_models import load_models
from utils.load_models import check_shared_state

def cli_loop():
    load_models()
    check_shared_state()
    
    print("\n🚀 Cyberbullying Detection CLI")
    print("Type your message and press Enter to get prediction. Type 'exit' to quit.\n")

    session_buffer = []
    SESSION_MAX_LENGTH = 5

    while True:
        text = input("Enter message: ").strip()
        if text.lower() in ['exit', 'quit']:
            print("Exiting CLI.")
            break
        if not text:
            print("Please enter some text.")
            continue

        try:
            # Single message prediction
            result = process_message(text)
            pred = result["prediction"]
            conf = result["confidence"]
            toxic_words = result.get("toxic_words", [])
            model_used = result.get("model", "unknown")
            
            explanation = (f"Potentially harmful words: {', '.join(toxic_words)}" 
                          if toxic_words else "No harmful words detected.")

            if pred == "toxic":
                print(f"\n🚨 Prediction: CYBERBULLYING (confidence {conf:.2%})")
                print(f"📊 Toxicity score: {result.get('toxicity_score', 0):.1f}%")
                print(f"🔍 Analysis: {explanation}")
            else:
                print(f"\n✅ Prediction: Not bullying (confidence {conf:.2%})")
                print(f"🔬 Model Used: {model_used}")
                print(f"📊 Toxicity score: {result.get('toxicity_score', 0):.1f}%")
                print(f"🔍 Analysis: {explanation}")
            print("-" * 50)
            # --- save to log ---
            with open("prediction_log.txt", "a", encoding="utf-8") as f:
                f.write(f"{text}\t{pred}\t{conf:.2f}\t{model_used}\n")

            # Update session buffer
            session_buffer.append({
                "text": text,
                "prediction": pred,
                "confidence": conf
            })
            if len(session_buffer) > SESSION_MAX_LENGTH:
                session_buffer.pop(0)

            if len(session_buffer) >= 3:
                # Count toxic messages in session
                toxic_msgs = [msg for msg in session_buffer if msg["prediction"] == "toxic"]
                toxic_count = len(toxic_msgs)

                if toxic_count >= 1:
                    print(f"🚨 Detected {toxic_count} toxic message(s) in recent conversation!")
                    print("🧾 Recent toxic content:")
                    for msg in toxic_msgs:
                        print(f"- {msg['text']} (confidence {msg['confidence']:.2%})")
                else:
                    print("✅ Normal conversation pattern detected")
                
                print("-" * 50)

        except Exception as e:
            print(f"\n⚠️ Error during prediction: {str(e)}")
            print("Please try again or report this issue.")
            print("-" * 50)

if __name__ == "__main__":
    cli_loop()
