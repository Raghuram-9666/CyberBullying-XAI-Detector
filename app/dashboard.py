import sys
import os
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.message_pipeline import process_message
from pipeline.session_pipeline import process_session
from main import load_models
from models.svm_model import custom_tokenizer

# Load models at startup
load_models()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "session_result" not in st.session_state:
    st.session_state.session_result = None

st.set_page_config(page_title="Cyberbullying Detector", layout="wide")
st.title("🛡️ Cyberbullying Detection with Explainable AI")

text = st.text_area("Enter your message:", height=150)
if st.button("Analyze Message"):
    if not text.strip():
        st.warning("Please enter a message")
    else:
        # Store message in session
        st.session_state.chat_history.append(text.strip())

        # Single Message Analysis (XLM-R or fallback)
        with st.spinner("Analyzing single message..."):
            single_result = process_message(text)
        
        st.subheader("🔍 Single Message Analysis")
        pred_text = "🚨 Cyberbullying Detected" if single_result["prediction"] == "toxic" else "✅ Safe Message"
        st.metric("Prediction", pred_text, f"Confidence: {single_result['confidence']:.2%}")

        # Display human-readable explanation (new)
        toxic_words = single_result.get("toxic_words", [])
        if toxic_words:
            st.markdown(f"⚠️ **Reason:** This message may be bullying because it uses words like: `{', '.join(toxic_words)}`")

        # Optional: Visual explanations
        with st.expander("Explainable AI Visuals"):
            try:
                st.pyplot(single_result["explanations"]["attention"])
                st.pyplot(single_result["explanations"]["ig"])
            except Exception:
                st.warning("Visual explanations not available.")

        # Session-based Analysis if ≥3 messages
        if len(st.session_state.chat_history) >= 3:
            with st.spinner("Analyzing session context..."):
                session_result = process_session(st.session_state.chat_history[-5:])
                st.session_state.session_result = session_result

            st.subheader("🧠 Session-Based Analysis")
            sess_pred = "🚨 Bullying Session" if session_result["prediction"] == "toxic" else "✅ Safe Session"
            st.metric("Session Prediction", sess_pred, f"Confidence: {session_result['confidence']:.2%}")
            st.pyplot(session_result["attention_plot"])

            # Top toxic messages (if any)
            if session_result.get("toxic_messages"):
                st.markdown("**Top Risky Messages:**")
                for msg, score in session_result["toxic_messages"]:
                    st.markdown(f"`{msg}` — Toxicity: `{score:.2f}`")

# Show Chat History
with st.expander("📜 Message History"):
    for i, msg in enumerate(st.session_state.chat_history[-10:], 1):
        st.markdown(f"{i}. {msg}")
