import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from pipeline.message_pipeline import process_message
from pipeline.session_pipeline import process_session
from models.svm_model import custom_tokenizer
from main import load_models  # Import load_models to load models once

# Load models once at app startup
load_models()

# --- Helper functions ---

def display_message_results(result):
    """Display single-message results"""
    st.subheader("🔍 Analysis Results")
    pred = result["prediction"]
    if isinstance(pred, str):
        pred_val = 1 if pred.lower() in ["toxic", "cyberbullying", "toxic"] else 0
    else:
        pred_val = pred
    pred_text = "Cyberbullying 🚨" if pred_val == 1 else "Non-bullying ✅"
    st.metric("Prediction", pred_text, f"Confidence: {result['confidence']:.2%}")

    st.subheader("🕵️‍♂️ Explanation")
    if result["model"] == "xlmr":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Attention Heatmap**")
            try:
                st.pyplot(result["explanations"]["attention"])
            except Exception as e:
                st.error(f"Attention plot error: {e}")
        with col2:
            st.markdown("**Integrated Gradients**")
            try:
                st.pyplot(result["explanations"]["ig"])
            except Exception as e:
                st.error(f"IG plot error: {e}")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**LIME Explanation (SVM)**")
            try:
                st.pyplot(result["explanations"]["lime"])
            except Exception as e:
                st.error(f"LIME plot error: {e}")
        with col2:
            st.markdown("**SHAP Explanation (LR)**")
            try:
                st.pyplot(result["explanations"]["shap"])
            except Exception as e:
                st.error(f"SHAP plot error: {e}")

def display_session_results(result):
    """Display session analysis results"""
    st.subheader("🔍 Session Analysis")
    pred = result["prediction"]
    if isinstance(pred, str):
        pred_val = 1 if pred.lower() in ["toxic", "cyberbullying", "toxic"] else 0
    else:
        pred_val = pred
    pred_text = "Cyberbullying Session 🚨" if pred_val == 1 else "Non-bullying Session ✅"
    st.metric("Session Prediction", pred_text, f"Confidence: {result['confidence']:.2%}")

    st.subheader("🧠 Contextual Attention")
    try:
        st.pyplot(result["attention_plot"])
    except Exception as e:
        st.error(f"Attention plot error: {e}")

    st.subheader("⚠️ Top Toxic Messages")
    toxic_msgs = result.get("toxic_messages", [])
    if toxic_msgs:
        for idx, msg, score in toxic_msgs:
            st.markdown(f"**Message #{idx+1}** (Toxicity: `{score:.2f}`)")
            st.code(msg, language="text")
    else:
        st.info("No toxic messages detected in the session.")

# --- Streamlit App ---

st.set_page_config(page_title="Cyberbullying Detector", layout="wide")
st.title("🛡️ Cyberbullying Detection with Explainable AI")

# Sidebar: User can add optional comments/notes about the input
st.sidebar.header("Add Comments / Notes")
user_comments = st.sidebar.text_area("Your comments or context (optional):", height=100)

# Input selection
input_type = st.radio("Select input type:", ("Single Message", "Message Session", "Bulk Upload"))

if input_type == "Single Message":
    text = st.text_area("Enter a message:", height=150)
    if st.button("Analyze Message"):
        if not text.strip():
            st.warning("Please enter a message")
        else:
            with st.spinner("Analyzing..."):
                result = process_message(text)
            display_message_results(result)
            if user_comments.strip():
                st.markdown(f"**Your Comments:** {user_comments.strip()}")

elif input_type == "Message Session":
    session = st.text_area("Enter messages (one per line):", height=200)
    if st.button("Analyze Session"):
        messages = [msg.strip() for msg in session.split('\n') if msg.strip()]
        if not messages:
            st.warning("Please enter at least one message")
        else:
            with st.spinner("Analyzing session..."):
                result = process_session(messages)
            display_session_results(result)
            if user_comments.strip():
                st.markdown(f"**Your Comments:** {user_comments.strip()}")

else:  # Bulk Upload
    st.info("Upload a plain text file with one message per line.")
    uploaded_file = st.file_uploader("Upload text file", type=["txt"])
    if uploaded_file is not None:
        content = uploaded_file.getvalue().decode("utf-8")
        messages = [line.strip() for line in content.split('\n') if line.strip()]
        st.write(f"Loaded {len(messages)} messages from file.")
        if st.button("Analyze Bulk Messages"):
            if not messages:
                st.warning("Uploaded file is empty or invalid.")
            else:
                all_results = []
                with st.spinner(f"Analyzing {len(messages)} messages..."):
                    for i, msg in enumerate(messages):
                        res = process_message(msg)
                        all_results.append((i, msg, res))
                st.subheader("Bulk Messages Analysis Results")
                for idx, msg, res in all_results:
                    st.markdown(f"### Message #{idx+1}")
                    st.write(msg)
                    pred = res["prediction"]
                    if isinstance(pred, str):
                        pred_val = 1 if pred.lower() in ["toxic", "cyberbullying", "toxic"] else 0
                    else:
                        pred_val = pred
                    pred_text = "Cyberbullying 🚨" if pred_val == 1 else "Non-bullying ✅"
                    st.metric("Prediction", pred_text, f"Confidence: {res['confidence']:.2%}")

                    # Show minimal explanations (only IG for speed)
                    if res["model"] == "xlmr":
                        st.markdown("**Integrated Gradients Explanation:**")
                        try:
                            st.pyplot(res["explanations"]["ig"])
                        except Exception as e:
                            st.error(f"IG plot error: {e}")
                    else:
                        st.markdown("Explanation plots not shown for bulk mode.")
                if user_comments.strip():
                    st.markdown(f"**Your Comments:** {user_comments.strip()}")
