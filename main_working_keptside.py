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
            # Single message prediction (handles emojis properly now)
            result = process_message(text)
            pred = result["prediction"]
            conf = result["confidence"]
            toxic_words = result.get("toxic_words", [])
            
            # Format explanation
            explanation = (f"Potentially harmful words: {', '.join(toxic_words)}" 
                          if toxic_words else "No harmful words detected.")

            # Print results
            if pred == "toxic":
                print(f"\n🚨 Prediction: CYBERBULLYING (confidence {conf:.2%})")
                print(f"📊 Toxicity score: {result.get('toxicity_score', 0):.1f}%")
                print(f"🔍 Analysis: {explanation}")
            else:
                print(f"\n✅ Prediction: Not bullying (confidence {conf:.2%})")
                print(f"📊 Toxicity score: {result.get('toxicity_score', 0):.1f}%")
                print(f"🔍 Analysis: {explanation}")
            print("-" * 50)

            # Session analysis
            session_buffer.append(text)
            if len(session_buffer) > SESSION_MAX_LENGTH:
                session_buffer.pop(0)

            if len(session_buffer) >= 3:
                sess_result = process_session(session_buffer)
                sess_pred = sess_result["prediction"]
                sess_conf = sess_result["confidence"]
                
                if sess_pred == "toxic":
                    print("🚨🚨🚨 CONSECUTIVE TOXIC MESSAGES DETECTED 🚨🚨🚨")
                    print(f"Session confidence: {sess_conf:.2%}")
                    print("Recent messages:")
                    for msg in session_buffer:
                        print(f"- {msg}")
                else:
                    print("✅ Normal conversation pattern detected")
                
                print("-" * 50)

        except Exception as e:
            print(f"\n⚠️ Error during prediction: {str(e)}")
            print("Please try again or report this issue.")
            print("-" * 50)

if __name__ == "__main__":
    cli_loop()