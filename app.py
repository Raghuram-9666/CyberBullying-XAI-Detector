import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import base64
from io import BytesIO
import re
import emoji
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import warnings
warnings.filterwarnings('ignore')
from collections import deque
from PIL import Image
import matplotlib.pyplot as plt
from utils.load_models import load_models
from pipeline.message_pipeline import process_message
from pipeline.session_pipeline import process_session
from pipeline import shared_state

# Global state for message history and session buffer
if 'message_history' not in st.session_state:
    st.session_state.message_history = deque(maxlen=3)
if 'session_buffer' not in st.session_state:
    st.session_state.session_buffer = []
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'admin_reports' not in st.session_state:
    st.session_state.admin_reports = []
if 'feedback' not in st.session_state:
    st.session_state.feedback = []

# Page Configuration
st.set_page_config(
    page_title="AI-Powered Cyberbullying Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models only once
if not shared_state.MODELS_LOADED:
    load_models()
    st.sidebar.success("✅ Models loaded successfully")

def load_css():
    """Apply custom CSS for professional UI"""
    dark_mode_css = """
    <style>
    .stApp { background-color: #2d2d2d; color: #e0e0e0; }
    .main-header { font-size: 3rem; font-weight: bold; text-align: center; background: linear-gradient(90deg, #4a5eb4 0%, #5e3d8a 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 2rem; text-shadow: 2px 2px 4px rgba(255,255,255,0.1); }
    .metric-card { background: linear-gradient(135deg, #4a5eb4 0%, #5e3d8a 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
    .toxic-alert { background: linear-gradient(135deg, #cc4c4c 0%, #bb3f3f 100%); padding: 1.5rem; border-radius: 15px; color: white; text-align: center; animation: pulse 2s infinite; border: 3px solid #ff5252; box-shadow: 0 8px 32px rgba(255, 107, 107, 0.4); }
    .safe-alert { background: linear-gradient(135deg, #3ea750 0%, #2f8a42 100%); padding: 1.5rem; border-radius: 15px; color: white; text-align: center; border: 3px solid #4caf50; box-shadow: 0 8px 32px rgba(76, 175, 80, 0.4); }
    .warning-alert { background: linear-gradient(135deg, #cca92e 0%, #d49e00 100%); padding: 1.5rem; border-radius: 15px; color: #fff; text-align: center; border: 3px solid #ff9800; box-shadow: 0 8px 32px rgba(255, 152, 0, 0.4); }
    @keyframes pulse { 0% { opacity: 1; transform: scale(1); } 50% { opacity: 0.8; transform: scale(1.02); } 100% { opacity: 1; transform: scale(1); } }
    .highlighted-toxic { background-color: #ff6b6b; color: white; padding: 3px 6px; border-radius: 5px; font-weight: bold; margin: 0 2px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); animation: highlight-flash 1s ease-in-out; }
    @keyframes highlight-flash { 0% { background-color: #ffeb3b; } 100% { background-color: #ff6b6b; } }
    .conversation-bubble { background: linear-gradient(135deg, #3d3d3d 0%, #4a4a4a 100%); padding: 15px; border-radius: 20px; margin: 10px 0; border-left: 5px solid #4a5eb4; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); transition: all 0.3s ease; color: #e0e0e0; }
    .conversation-bubble:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15); }
    .toxic-bubble { background: linear-gradient(135deg, #4a2d2d 0%, #5e3d3d 100%); border-left: 5px solid #cc4c4c; animation: warning-glow 2s ease-in-out infinite; }
    @keyframes warning-glow { 0%, 100% { box-shadow: 0 4px 6px rgba(244, 67, 54, 0.3); } 50% { box-shadow: 0 4px 20px rgba(244, 67, 54, 0.6); } }
    .model-confidence { background: linear-gradient(135deg, #1a2d4a 0%, #2d507a 100%); padding: 15px; border-radius: 12px; border: 2px solid #1e6091; margin: 8px 0; transition: all 0.3s ease; color: #e0e0e0; }
    .model-confidence:hover { transform: scale(1.02); box-shadow: 0 6px 20px rgba(33, 150, 243, 0.3); }
    .tag-badge { background: linear-gradient(135deg, #4a5eb4 0%, #5e3d8a 100%); color: white; padding: 6px 14px; border-radius: 25px; margin: 3px; display: inline-block; font-size: 0.85em; font-weight: bold; box-shadow: 0 2px 8px rgba(0,0,0,0.2); transition: all 0.2s ease; }
    .tag-badge:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
    .sidebar .sidebar-content { background: linear-gradient(180deg, #4a5eb4 0%, #5e3d8a 100%); color: #e0e0e0; }
    .stSelectbox > div > div { background-color: #3d3d3d; border: 2px solid #4a5eb4; border-radius: 8px; transition: all 0.3s ease; color: #e0e0e0; }
    .stSelectbox > div > div:hover { border-color: #667eea; box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1); }
    .explanation-box { background: linear-gradient(135deg, #4a2d1a 0%, #7a4d2d 100%); padding: 15px; border-radius: 10px; border-left: 5px solid #d49e00; margin: 10px 0; color: #e0e0e0; }
    .progress-container { background: #3d3d3d; border-radius: 20px; padding: 3px; margin: 10px 0; }
    .progress-bar { height: 20px; border-radius: 20px; transition: width 0.5s ease; }
    .progress-toxic { background: linear-gradient(90deg, #ff6b6b 0%, #ee5a52 100%); }
    .progress-safe { background: linear-gradient(90deg, #51cf66 0%, #40c057 100%); }
    .stTextArea textarea { border-radius: 1rem; font-size: 16px; padding: 1rem; background-color: #3d3d3d; color: #e0e0e0; border: 2px solid #4a5eb4; }
    .stButton > button { background-color: #4a5eb4; color: white; border-radius: 8px; }
    .stButton > button:hover { background-color: #667eea; }
    .demo-image-container { width: 100%; display: flex; justify-content: center; align-items: center; margin: 20px auto; text-align: center; }
    .demo-image-container img { border-radius: 10px; border: 2px solid #4a5eb4; box-shadow: 0 4px 8px rgba(0,0,0,0.1); transition: transform 0.2s ease; margin: 0 auto; }
    .demo-image-container img:hover { transform: scale(1.05); }
    .demo-caption { text-align: center; font-size: 0.9em; color: #b0b0b0; margin-top: 8px; }
    </style>
    """
    light_mode_css = """
    <style>
    .stApp { background-color: #fafafa; color: #333; }
    .main-header { font-size: 3rem; font-weight: bold; text-align: center; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 2rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
    .toxic-alert { background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%); padding: 1.5rem; border-radius: 15px; color: white; text-align: center; animation: pulse 2s infinite; border: 3px solid #ff5252; box-shadow: 0 8px 32px rgba(255, 107, 107, 0.4); }
    .safe-alert { background: linear-gradient(135deg, #51cf66 0%, #40c057 100%); padding: 1.5rem; border-radius: 15px; color: white; text-align: center; border: 3px solid #4caf50; box-shadow: 0 8px 32px rgba(76, 175, 80, 0.4); }
    .warning-alert { background: linear-gradient(135deg, #ffd43b 0%, #fab005 100%); padding: 1.5rem; border-radius: 15px; color: #333; text-align: center; border: 3px solid #ff9800; box-shadow: 0 8px 32px rgba(255, 152, 0, 0.4); }
    @keyframes pulse { 0% { opacity: 1; transform: scale(1); } 50% { opacity: 0.8; transform: scale(1.02); } 100% { opacity: 1; transform: scale(1); } }
    .highlighted-toxic { background-color: #ff6b6b; color: white; padding: 3px 6px; border-radius: 5px; font-weight: bold; margin: 0 2px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); animation: highlight-flash 1s ease-in-out; }
    @keyframes highlight-flash { 0% { background-color: #ffeb3b; } 100% { background-color: #ff6b6b; } }
    .conversation-bubble { background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 15px; border-radius: 20px; margin: 10px 0; border-left: 5px solid #667eea; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); transition: all 0.3s ease; }
    .conversation-bubble:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15); }
    .toxic-bubble { background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%); border-left: 5px solid #f44336; animation: warning-glow 2s ease-in-out infinite; }
    @keyframes warning-glow { 0%, 100% { box-shadow: 0 4px 6px rgba(244, 67, 54, 0.3); } 50% { box-shadow: 0 4px 20px rgba(244, 67, 54, 0.6); } }
    .model-confidence { background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 15px; border-radius: 12px; border: 2px solid #2196f3; margin: 8px 0; transition: all 0.3s ease; }
    .model-confidence:hover { transform: scale(1.02); box-shadow: 0 6px 20px rgba(33, 150, 243, 0.3); }
    .tag-badge { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 6px 14px; border-radius: 25px; margin: 3px; display: inline-block; font-size: 0.85em; font-weight: bold; box-shadow: 0 2px 8px rgba(0,0,0,0.2); transition: all 0.2s ease; }
    .tag-badge:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
    .sidebar .sidebar-content { background: linear-gradient(180deg, #667eea 0%, #764ba2 100%); }
    .stSelectbox > div > div { background-color: #f8f9fa; border: 2px solid #e9ecef; border-radius: 8px; transition: all 0.3s ease; }
    .stSelectbox > div > div:hover { border-color: #667eea; box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1); }
    .explanation-box { background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); padding: 15px; border-radius: 10px; border-left: 5px solid #ff9800; margin: 10px 0; }
    .progress-container { background: #f5f5f5; border-radius: 20px; padding: 3px; margin: 10px 0; }
    .progress-bar { height: 20px; border-radius: 20px; transition: width 0.5s ease; }
    .progress-toxic { background: linear-gradient(90deg, #ff6b6b 0%, #ee5a52 100%); }
    .progress-safe { background: linear-gradient(90deg, #51cf66 0%, #40c057 100%); }
    .stTextArea textarea { border-radius: 1rem; font-size: 16px; padding: 1rem; background-color: #ffffff; color: #333; border: 2px solid #e9ecef; }
    .stButton > button { background-color: #667eea; color: white; border-radius: 8px; }
    .stButton > button:hover { background-color: #764ba2; }
    .demo-image-container { width: 100%; display: flex; justify-content: center; align-items: center; margin: 20px auto; text-align: center; }
    .demo-image-container img { border-radius: 10px; border: 2px solid #e9ecef; box-shadow: 0 4px 8px rgba(0,0,0,0.1); transition: transform 0.2s ease; margin: 0 auto; }
    .demo-image-container img:hover { transform: scale(1.05); }
    .demo-caption { text-align: center; font-size: 0.9em; color: #666; margin-top: 8px; }
    </style>
    """
    css = dark_mode_css if st.session_state.get('dark_mode', False) else light_mode_css
    st.markdown(css, unsafe_allow_html=True)

def init_session_state():
    if 'conversation_history' not in st.session_state: st.session_state.conversation_history = []
    if 'flagged_comments' not in st.session_state: st.session_state.flagged_comments = []
    if 'dark_mode' not in st.session_state: st.session_state.dark_mode = False
    if 'session_stats' not in st.session_state: st.session_state.session_stats = {'total_messages': 0, 'toxic_messages': 0, 'safe_messages': 0, 'average_toxicity': 0.0}
    if 'admin_reports' not in st.session_state: st.session_state.admin_reports = []
    if 'feedback' not in st.session_state: st.session_state.feedback = []

def extract_toxic_words(text):
    """Extract potentially toxic words from text"""
    toxic_keywords = ['fuck', 'bitch', 'ass', 'damn', 'shit']
    words = re.findall(r'\b\w+\b', text.lower())
    found_toxic = [word for word in words for toxic in toxic_keywords if toxic in word or word in toxic]
    return list(set(found_toxic))

def highlight_toxic_words(text, toxic_words):
    """Highlight toxic words in text"""
    if not toxic_words: return text
    highlighted = text
    for word in toxic_words:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        highlighted = pattern.sub(f'<span class="highlighted-toxic">{word}</span>', highlighted)
    return highlighted

def get_language_tags(text):
    """Analyze text and return language/tone tags"""
    tags = []
    emoji_count = len(re.findall(r'[😀-🙏]', text))
    if emoji_count > 0: tags.append(f"emoji-rich ({emoji_count})")
    caps_count = sum(1 for c in text if c.isupper())
    if caps_count > len(text) * 0.5 and len(text) > 10: tags.append("aggressive-tone")
    word_count = len(text.split())
    if word_count > 30: tags.append("lengthy")
    elif word_count < 5: tags.append("brief")
    else: tags.append("moderate")
    if exclamation_count := text.count('!') > 2: tags.append("emphatic")
    if question_count := text.count('?') > 1: tags.append("inquisitive")
    if re.search(r'\b(the|and|is|are|you|your)\b', text.lower()): tags.append("english")
    return tags

def get_sentiment_score(text):
    """Simple sentiment analysis"""
    positive_words = ['good', 'great', 'awesome', 'nice', 'love', 'like', 'happy', 'excellent']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 'horrible', 'fuck', 'bitch']
    words = text.lower().split()
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    total_words = len(words)
    return 0.0 if total_words == 0 else max(-1.0, min(1.0, (positive_count - negative_count) / total_words))

def generate_pdf_report(conversation_history, flagged_comments):
    """Generate comprehensive PDF report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    title = Paragraph("🛡️ Cyberbullying Detection System - Analysis Report", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 20))
    total_messages = len(conversation_history)
    toxic_messages = len([msg for msg in conversation_history if msg['status'] == 'toxic'])
    safety_score = ((total_messages - toxic_messages) / max(total_messages, 1) * 100)
    summary_data = [['Metric', 'Value'], ['Report Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')], ['Total Messages Analyzed', str(total_messages)], ['Toxic Messages Detected', str(toxic_messages)], ['Safety Score', f"{safety_score:.1f}%"], ['Flagged for Review', str(len(flagged_comments))]]
    summary_table = Table(summary_data)
    summary_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey), ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke), ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, 0), 14), ('BOTTOMPADDING', (0, 0), (-1, 0), 12), ('BACKGROUND', (0, 1), (-1, -1), colors.beige), ('GRID', (0, 0), (-1, -1), 1, colors.black)]))
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    story.append(summary_table)
    story.append(Spacer(1, 20))
    if conversation_history:
        story.append(Paragraph("Detailed Message Analysis", styles['Heading2']))
        story.append(Spacer(1, 12))
        for i, msg in enumerate(conversation_history, 1):
            status_color = colors.red if msg['status'] == 'toxic' else colors.green
            msg_text = f"<b>Message {i}:</b><br/><b>Content:</b> {msg['text'][:200]}{'...' if len(msg['text']) > 200 else ''}<br/><b>Status:</b> <font color='{('red' if msg['status'] == 'toxic' else 'green')}'>{msg['status'].upper()}</font><br/><b>Confidence:</b> {msg['confidence']:.2%}<br/><b>Toxicity Score:</b> {msg['toxicity_score']:.1f}%<br/><b>Models Used:</b> {', '.join(msg.get('models_used', ['Unknown']))}<br/><b>Timestamp:</b> {msg['timestamp']}<br/>"
            story.append(Paragraph(msg_text, styles['Normal']))
            story.append(Spacer(1, 10))
    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_flagged_report(flagged_comments):
    """Generate PDF report for flagged comments"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    title = Paragraph("🛡️ Flagged Comments Report", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 20))
    if flagged_comments:
        story.append(Paragraph("Flagged Comments Details", styles['Heading2']))
        story.append(Spacer(1, 12))
        for i, flag in enumerate(flagged_comments, 1):
            flag_text = f"<b>Flagged Comment {i}:</b><br/><b>Content:</b> {flag['text'][:200]}{'...' if len(flag['text']) > 200 else ''}<br/><b>Reason:</b> {flag['reason']}<br/><b>Toxicity Score:</b> {flag['toxicity_score']:.1f}%<br/><b>Timestamp:</b> {flag['timestamp']}<br/>"
            story.append(Paragraph(flag_text, styles['Normal']))
            story.append(Spacer(1, 10))
    else:
        story.append(Paragraph("No flagged comments available.", styles['Normal']))
    doc.build(story)
    buffer.seek(0)
    return buffer

def analyze_text(text, active_models):
    """Analyze text using process_message and process_session"""
    results = {}
    
    # Store message for session analysis
    st.session_state.message_history.append(text)
    st.session_state.session_buffer.append(text)
    if len(st.session_state.session_buffer) > 5:
        st.session_state.session_buffer.pop(0)
    
    # Process single message
    result = process_message(text)
    results['primary'] = result
    final_prediction = result.get('prediction', 'error')
    confidence = result.get('confidence', 0)
    toxicity_score = result.get('toxicity_score', 0)
    toxic_words = result.get('toxic_words', [])
    model_used = result.get('model', 'Unknown')
    
    # Process session if BiLSTM is active and enough messages exist
    if 'BiLSTM' in active_models and len(st.session_state.session_buffer) >= 3:
        session_result = process_session(st.session_state.session_buffer)
        results['bilstm_session'] = session_result
    
    return {
        'final_prediction': final_prediction,
        'confidence': confidence,
        'toxicity_score': toxicity_score,
        'model': model_used,
        'toxic_words': toxic_words,
        'explanations': result.get('explanations', {})
    }

def translate_text(text, language):
    """Translate text using a static dictionary"""
    translations = {
        "English": {
            "main_header": "🛡️ Cyberbullying Detection System with Explainable AI",
            "message_interface": "💬 Message Analysis Interface",
            "conversation_thread": "💬 Conversation Thread",
            "session_analytics": "📊 Session Analytics",
            "generate_report": "📄 Generate Report",
            "no_conversation": "🤖 No conversation history yet. Start analyzing messages to see the thread!",
            "no_data": "📝 No data available for report generation",
            "model_performance": "🎯 AI Model Performance Analysis",
            "toxic_analysis": "🔍 Toxic Content Analysis",
            "content_classification": "🏷️ Content Classification",
            "toxicity_breakdown": "📊 Toxicity Breakdown",
            "content_moderation": "🚩 Content Moderation",
            "model_explainability": "🔬 AI Model Explainability",
            "model_decision": "Model Decision Analysis",
            "model_breakdown": "Model Decision Breakdown"
        },
        "Spanish": {
            "main_header": "🛡️ Sistema de Detección de Ciberacoso con IA Explicable",
            "message_interface": "💬 Interfaz de Análisis de Mensajes",
            "conversation_thread": "💬 Hilo de Conversación",
            "session_analytics": "📊 Análisis de Sesión",
            "generate_report": "📄 Generar Informe",
            "no_conversation": "🤖 ¡Aún no hay historial de conversación! Comienza a analizar mensajes para ver el hilo.",
            "no_data": "📝 No hay datos disponibles para la generación de informes",
            "model_performance": "🎯 Análisis de Rendimiento del Modelo de IA",
            "toxic_analysis": "🔍 Análisis de Contenido Tóxico",
            "content_classification": "🏷️ Clasificación de Contenido",
            "toxicity_breakdown": "📊 Desglose de Toxicidad",
            "content_moderation": "🚩 Moderación de Contenido",
            "model_explainability": "🔬 Explicabilidad del Modelo de IA",
            "model_decision": "Análisis de Decisiones del Modelo",
            "model_breakdown": "Desglose de Decisiones del Modelo"
        },
        "French": {
            "main_header": "🛡️ Système de Détection de Cyberharcèlement avec IA Explicable",
            "message_interface": "💬 Interface d'Analyse de Messages",
            "conversation_thread": "💬 Fil de Conversation",
            "session_analytics": "📊 Analyse de Session",
            "generate_report": "📄 Générer un Rapport",
            "no_conversation": "🤖 Aucun historique de conversation pour l'instant. Commencez à analyser des messages pour voir le fil !",
            "no_data": "📝 Aucune donnée disponible pour la génération de rapports",
            "model_performance": "🎯 Analyse des Performances du Modèle IA",
            "toxic_analysis": "🔍 Analyse de Contenu Toxique",
            "content_classification": "🏷️ Classification de Contenu",
            "toxicity_breakdown": "📊 Répartition de la Toxicité",
            "content_moderation": "🚩 Modération de Contenu",
            "model_explainability": "🔬 Explicabilité du Modèle IA",
            "model_decision": "Analyse des Décisions du Modèle",
            "model_breakdown": "Répartition des Décisions du Modèle"
        },
        "German": {
            "main_header": "🛡️ Cybermobbing-Erkennungssystem mit erklärbarer KI",
            "message_interface": "💬 Schnittstelle zur Nachrichtenanalyse",
            "conversation_thread": "💬 Gesprächsverlauf",
            "session_analytics": "📊 Sitzungsanalyse",
            "generate_report": "📄 Bericht generieren",
            "no_conversation": "🤖 Noch kein Gesprächsverlauf. Beginnen Sie mit der Analyse von Nachrichten, um den Verlauf zu sehen!",
            "no_data": "📝 Keine Daten für die Berichtserstellung verfügbar",
            "model_performance": "🎯 Leistungsanalyse des KI-Modells",
            "toxic_analysis": "🔍 Analyse toxischer Inhalte",
            "content_classification": "🏷️ Inhaltsklassifizierung",
            "toxicity_breakdown": "📊 Aufschlüsselung der Toxizität",
            "content_moderation": "🚩 Inhaltsmoderation",
            "model_explainability": "🔬 Erklärbarkeit des KI-Modells",
            "model_decision": "Analyse der Modellentscheidungen",
            "model_breakdown": "Aufschlüsselung der Modellentscheidungen"
        },
        "Italian": {
            "main_header": "🛡️ Sistema di Rilevamento del Cyberbullismo con IA Esplicabile",
            "message_interface": "💬 Interfaccia di Analisi dei Messaggi",
            "conversation_thread": "💬 Filo di Conversazione",
            "session_analytics": "📊 Analisi della Sessione",
            "generate_report": "📄 Genera Rapporto",
            "no_conversation": "🤖 Nessuna cronologia di conversazione. Inizia ad analizzare i messaggi per vedere il filo!",
            "no_data": "📝 Nessun dato disponibile per la generazione del rapporto",
            "model_performance": "🎯 Analisi delle Prestazioni del Modello IA",
            "toxic_analysis": "🔍 Analisi dei Contenuti Tossici",
            "content_classification": "🏷️ Classificazione dei Contenuti",
            "toxicity_breakdown": "📊 Ripartizione della Tossicità",
            "content_moderation": "🚩 Moderazione dei Contenuti",
            "model_explainability": "🔬 Spiegabilità del Modello IA",
            "model_decision": "Analisi delle Decisioni del Modello",
            "model_breakdown": "Ripartizione delle Decisioni del Modello"
        },
        "Portuguese": {
            "main_header": "🛡️ Sistema de Detecção de Cyberbullying com IA Explicável",
            "message_interface": "💬 Interface de Análise de Mensagens",
            "conversation_thread": "💬 Fio de Conversa",
            "session_analytics": "📊 Análise de Sessão",
            "generate_report": "📄 Gerar Relatório",
            "no_conversation": "🤖 Ainda não há histórico de conversas. Comece a analisar mensagens para ver o fio!",
            "no_data": "📝 Nenhum dado disponível para geração de relatórios",
            "model_performance": "🎯 Análise de Desempenho do Modelo de IA",
            "toxic_analysis": "🔍 Análise de Conteúdo Tóxico",
            "content_classification": "🏷️ Classificação de Conteúdo",
            "toxicity_breakdown": "📊 Detalhamento da Toxicidade",
            "content_moderation": "🚩 Moderação de Conteúdo",
            "model_explainability": "🔬 Explicabilidade do Modelo de IA",
            "model_decision": "Análise das Decisões do Modelo",
            "model_breakdown": "Detalhamento das Decisões do Modelo"
        },
        "Arabic": {
            "main_header": "🛡️ نظام كشف التنمر الإلكتروني باستخدام الذكاء الاصطناعي القابل للتفسير",
            "message_interface": "💬 واجهة تحليل الرسائل",
            "conversation_thread": "💬 سلسلة المحادثة",
            "session_analytics": "📊 تحليل الجلسة",
            "generate_report": "📄 إنشاء تقرير",
            "no_conversation": "🤖 لا يوجد سجل محادثات بعد. ابدأ بتحليل الرسائل لعرض السلسلة!",
            "no_data": "📝 لا توجد بيانات متاحة لإنشاء التقرير",
            "model_performance": "🎯 تحليل أداء نموذج الذكاء الاصطناعي",
            "toxic_analysis": "🔍 تحليل المحتوى السام",
            "content_classification": "🏷️ تصنيف المحتوى",
            "toxicity_breakdown": "📊 تفصيل السمية",
            "content_moderation": "🚩 الإشراف على المحتوى",
            "model_explainability": "🔬 قابلية تفسير نموذج الذكاء الاصطناعي",
            "model_decision": "تحليل قرارات النموذج",
            "model_breakdown": "تفصيل قرارات النموذج"
        },
        "Chinese": {
            "main_header": "🛡️ 使用可解释人工智能的网络欺凌检测系统",
            "message_interface": "💬 消息分析界面",
            "conversation_thread": "💬 对话线程",
            "session_analytics": "📊 会话分析",
            "generate_report": "📄 生成报告",
            "no_conversation": "🤖 尚无对话历史记录。开始分析消息以查看线程！",
            "no_data": "📝 没有可用于生成报告的数据",
            "model_performance": "🎯 人工智能模型性能分析",
            "toxic_analysis": "🔍 有毒内容分析",
            "content_classification": "🏷️ 内容分类",
            "toxicity_breakdown": "📊 毒性分解",
            "content_moderation": "🚩 内容审核",
            "model_explainability": "🔬 人工智能模型可解释性",
            "model_decision": "模型决策分析",
            "model_breakdown": "模型决策分解"
        }
    }
    return translations.get(language, translations["English"]).get(text, text)

def main():
    load_css()
    init_session_state()
    
    # Language-based UI text
    language = st.session_state.get('language', 'English')
    st.markdown(f'<h1 class="main-header">{translate_text("main_header", language)}</h1>', unsafe_allow_html=True)
    
    # Display demo post
    try:
        demo_image = Image.open("assets/demo_post.png")
        st.markdown('<div class="demo-image-container">', unsafe_allow_html=True)
        st.image(demo_image, use_container_width=False, output_format="auto", width=1000, caption="Sample Instagram-style post")
    except FileNotFoundError:
        st.warning("⚠️ Demo post image not found. Please ensure 'assets/demo_post.png' exists.")
    
    with st.sidebar:
        st.header("⚙️ System Configuration")
        dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode, key="dark_mode_toggle")
        if dark_mode != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_mode
            st.rerun()
        toxicity_threshold = st.slider("Toxicity Threshold", 0.0, 1.0, 0.75, 0.05)
        tone_sensitivity = st.slider("Tone Sensitivity", 0.0, 1.0, 0.7, 0.1)
        selected_models = st.multiselect(
            "Active Models:",
            ["XLM-RoBERTa", "BiLSTM", "SVM", "Logistic Regression"],
            default=["XLM-RoBERTa", "BiLSTM", "SVM", "Logistic Regression"]
        )
        language = st.selectbox(
            "Interface Language:",
            ["English", "Spanish", "French", "German", "Italian", "Portuguese", "Arabic", "Chinese"],
            index=["English", "Spanish", "French", "German", "Italian", "Portuguese", "Arabic", "Chinese"].index(st.session_state.get('language', 'English')),
            key="language_select"
        )
        if language != st.session_state.get('language', 'English'):
            st.session_state.language = language
            st.rerun()
        st.subheader("📊 System Status")
        st.metric("Active Models", len(selected_models))
        st.metric("Session Messages", len(st.session_state.conversation_history))
        st.metric("Flagged Comments", len(st.session_state.flagged_comments))
        if st.button("🗑️ Clear Session", type="secondary"):
            st.session_state.conversation_history = []
            st.session_state.flagged_comments = []
            st.session_state.session_stats = {
                'total_messages': 0, 'toxic_messages': 0, 
                'safe_messages': 0, 'average_toxicity': 0.0
            }
            st.session_state.session_buffer = []
            st.session_state.feedback = []
            st.rerun()
    
    main_col1, main_col2 = st.columns([2, 1])
    with main_col1:
        st.subheader(translate_text("message_interface", language))
        uploaded_file = st.file_uploader("Upload text file", type=['txt', 'csv'])
        input_method = st.radio("Input Method:", ["Text Input", "File Upload"], horizontal=True)
        user_input = st.text_area(
            "Enter your message (supports emojis and multiple languages):",
            height=120,
            placeholder="Type your thoughts here... 😊 Supports multilingual content!",
            help="This system can analyze text in multiple languages and understands emoji context."
        ) if input_method == "Text Input" else (str(uploaded_file.read(), "utf-8") if uploaded_file else "")
        if uploaded_file:
            st.text_area("File Content:", user_input, height=120, disabled=True)
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn1: analyze_button = st.button("🔍 Analyze Comment", type="primary", use_container_width=True)
        with col_btn2: batch_analyze = st.button("📊 Batch Analysis", use_container_width=True)
        with col_btn3: real_time = st.toggle(translate_text("real_time_mode", language) if 'real_time_mode' in translate_text("", language) else "⚡ Real-time Mode")
        if real_time and user_input and len(user_input) > 10: 
            analyze_button = True
            st.rerun()
        if analyze_button and user_input.strip():
            with st.spinner("🤖 AI Models Processing..."):
                result = analyze_text(user_input, selected_models)
                st.session_state.last_result = result
                
                if result['final_prediction'] != 'error':
                    final_prediction = result['final_prediction']
                    avg_confidence = result['confidence']
                    avg_toxicity = result['toxicity_score']
                    toxic_words = result.get('toxic_words', [])
                    model_used = result['model']
                    
                    with st.chat_message("assistant"):
                        col1, col2 = st.columns([8, 2])
                        with col1:
                            st.markdown(f"**🧾 Prediction**: {'🚨 Toxic' if final_prediction == 'toxic' else '✅ Non-Toxic'}")
                            st.markdown(f"**📊 Confidence**: `{avg_confidence:.2%}`")
                            st.markdown(f"**🔬 Model Used**: `{model_used}`")
                            if toxic_words:
                                st.markdown(f"**🚩 Toxic Words**: `{', '.join(toxic_words)}`")
                            st.markdown(f"**📈 Toxicity Score**: `{avg_toxicity:.1f}%`")
                        with col2:
                            if final_prediction == "toxic":
                                st.error("Toxic")
                            else:
                                st.success("Clean")
                        
                        with st.expander("📉 Explanation Visuals", expanded=False):
                            for k, v in result.get("explanations", {}).items():
                                if hasattr(v, 'savefig'):
                                    st.pyplot(v)
                                elif isinstance(v, str):
                                    st.markdown(v)
                    
                    if 'BiLSTM' in selected_models and len(st.session_state.session_buffer) >= 3:
                        sess_result = process_session(st.session_state.session_buffer)
                        with st.expander("🧠 Session Tone (BiLSTM)", expanded=False):
                            st.markdown(f"**🗣️ Tone**: `{'🚨 Toxic' if sess_result['prediction'] == 'toxic' else '✅ Normal'}`")
                            st.markdown(f"**📊 Confidence**: `{sess_result['confidence']:.2%}`")
                            fig = sess_result.get("attention_plot")
                            if fig:
                                st.pyplot(fig)
                            else:
                                st.info("No attention visualization available for this session.")
                    
                    if final_prediction == 'toxic' and avg_toxicity > toxicity_threshold * 100:
                        st.markdown(f'<div class="toxic-alert"><h2>🚨 HIGH TOXICITY DETECTED</h2><h3>Cyberbullying Risk: CRITICAL</h3><p><strong>Toxicity Score:</strong> {avg_toxicity:.1f}%</p><p><strong>Model Consensus:</strong> {avg_confidence:.1%}</p><p><strong>Threat Level:</strong> {"🔴 SEVERE" if avg_toxicity > 80 else "🟠 MODERATE"}</p></div>', unsafe_allow_html=True)
                        st.session_state.flagged_comments.append({'text': user_input, 'toxicity_score': avg_toxicity, 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'reason': 'Auto-flagged: High toxicity score'})
                        st.session_state.feedback.append(f"⚠️ Your message has been flagged (Auto-flagged)")
                    elif avg_toxicity > (toxicity_threshold * 100 * 0.6):
                        st.markdown(f'<div class="warning-alert"><h3>⚠️ POTENTIALLY HARMFUL CONTENT</h3><p><strong>Toxicity Score:</strong> {avg_toxicity:.1f}%</p><p><strong>Confidence:</strong> {avg_confidence:.1%}</p><p><strong>Status:</strong> Requires Review</p></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="safe-alert"><h3>✅ CONTENT APPROVED</h3><p><strong>Safety Score:</strong> {100 - avg_toxicity:.1f}%</p><p><strong>Confidence:</strong> {avg_confidence:.1%}</p><p><strong>Status:</strong> Safe for Community</p></div>', unsafe_allow_html=True)
                    
                    st.subheader(translate_text("model_performance", language))
                    st.markdown(f'<div class="model-confidence"><h4>🟢 {model_used}</h4><p><strong>Confidence:</strong> {avg_confidence:.1%}</p><p><strong>Toxicity:</strong> {avg_toxicity:.1f}%</p><p><strong>Vote:</strong> {final_prediction.upper()}</p></div>', unsafe_allow_html=True)
                    
                    if toxic_words:
                        st.subheader(translate_text("toxic_analysis", language))
                        highlighted_text = highlight_toxic_words(user_input, toxic_words)
                        st.markdown(f'<div class="explanation-box"><h4>Highlighted Analysis:</h4><p>{highlighted_text}</p><p><strong>Flagged Terms:</strong> {", ".join(toxic_words)}</p></div>', unsafe_allow_html=True)
                    
                    st.subheader(translate_text("content_classification", language))
                    tags = get_language_tags(user_input)
                    tag_html = "".join(f'<span class="tag-badge">{tag}</span>' for tag in tags)
                    st.markdown(tag_html, unsafe_allow_html=True)
                    sentiment = get_sentiment_score(user_input)
                    sentiment_label = "😊 Positive" if sentiment > 0.1 else "😐 Neutral" if sentiment > -0.1 else "😔 Negative"
                    col_sent1, col_sent2 = st.columns(2)
                    with col_sent1: st.metric("Sentiment Analysis", sentiment_label, f"{sentiment:.2f}")
                    with col_sent2:
                        emoji_count = len(re.findall(r'[😀-🙏]', user_input))
                        if emoji_count > 0: st.metric("Emoji Analysis", f"{emoji_count} detected", "Context-aware")
                    
                    st.subheader(translate_text("toxicity_breakdown", language))
                    progress_html = f'<div class="progress-container"><div class="progress-bar {("progress-toxic" if avg_toxicity > 50 else "progress-safe")}" style="width: {avg_toxicity}%;">{avg_toxicity:.1f}% Toxicity</div></div>'
                    st.markdown(progress_html, unsafe_allow_html=True)
                    
                    message_data = {
                        'text': user_input,
                        'status': final_prediction,
                        'confidence': avg_confidence,
                        'toxicity_score': avg_toxicity,
                        'sentiment': sentiment,
                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                        'models_used': [model_used],
                        'toxic_words': toxic_words,
                        'tags': tags
                    }
                    st.session_state.conversation_history.append(message_data)
                    st.session_state.session_stats['total_messages'] += 1
                    if final_prediction == 'toxic': st.session_state.session_stats['toxic_messages'] += 1
                    else: st.session_state.session_stats['safe_messages'] += 1
                    
                    if final_prediction == 'toxic' or avg_toxicity > toxicity_threshold * 100:
                        st.subheader(translate_text("content_moderation", language))
                        col_flag1, col_flag2, col_flag3 = st.columns(3)
                        with col_flag1: 
                            if st.button("🚩 Flag Comment", type="secondary"): 
                                flag_data = {'text': user_input, 'toxicity_score': avg_toxicity, 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'reason': 'User flagged', 'models_consensus': model_used}
                                st.session_state.flagged_comments.append(flag_data)
                                st.session_state.feedback.append(f"⚠️ Your message has been flagged (Manual)")
                        with col_flag2: 
                            if st.button("📧 Report to Admin"): 
                                admin_data = {'text': user_input, 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'reason': 'User reported to admin'}
                                st.session_state.admin_reports.append(admin_data)
                        with col_flag3: 
                            if st.button("🔇 Auto-moderate"): 
                                for i, msg in enumerate(st.session_state.conversation_history):
                                    if msg['text'] == user_input:
                                        st.session_state.conversation_history.pop(i)
                                        st.session_state.feedback.append(f"✅ You removed this message")
                                        break
                    
                    with st.expander(translate_text("model_explainability", language), expanded=False):
                        st.subheader(translate_text("model_decision", language))
                        explanation_data = [{
                            'Model': model_used,
                            'Confidence': avg_confidence,
                            'Toxicity Score': avg_toxicity,
                            'Decision': final_prediction
                        }]
                        exp_df = pd.DataFrame(explanation_data)
                        fig = px.bar(exp_df, x='Model', y='Toxicity Score', color='Decision', color_discrete_map={'toxic': '#ff6b6b', 'safe': '#51cf66'}, title=translate_text("model_breakdown", language))
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        factors = []
                        if toxic_words: factors.append(f"🔴 Toxic vocabulary: {len(toxic_words)} words detected")
                        if emoji_count > 0: factors.append(f"😊 Emoji context: {emoji_count} emojis analyzed")
                        if len(user_input.split()) > 20: factors.append("📝 Message length: Extended content")
                        if sentiment < -0.2: factors.append("😔 Negative sentiment detected")
                        for factor in factors: st.write(f"• {factor}")
                else:
                    st.error("❌ Model failed to process the input. Please check model availability.")
    
    with main_col2:
        st.subheader(translate_text("conversation_thread", language))
        if st.session_state.conversation_history:
            filter_option = st.selectbox("Filter Messages:", ["All", "Toxic Only", "Safe Only", "Recent 10"])
            filtered_history = st.session_state.conversation_history.copy()
            if filter_option == "Toxic Only": filtered_history = [msg for msg in filtered_history if msg['status'] == 'toxic']
            elif filter_option == "Safe Only": filtered_history = [msg for msg in filtered_history if msg['status'] == 'safe']
            elif filter_option == "Recent 10": filtered_history = filtered_history[-10:]
            for i, msg in enumerate(reversed(filtered_history)):
                bubble_class = "toxic-bubble" if msg['status'] == 'toxic' else ""
                risk_emoji = "🔴" if msg['toxicity_score'] > 75 else "🟡" if msg['toxicity_score'] > 50 else "🟢"
                flagged_html = "<small style='background: #ffebee; padding: 2px 6px; border-radius: 10px;'>⚠️ Flagged</small>" if msg.get("toxic_words") else ""
                st.markdown(f'<div class="conversation-bubble {bubble_class}"><div style="display: flex; justify-content: space-between; align-items: center;"><strong>{risk_emoji} {msg["timestamp"]}</strong><span style="font-size: 0.8em; color: #666;">{msg["toxicity_score"]:.1f}% | {msg["confidence"]:.0%}</span></div><p style="margin: 8px 0;">{msg["text"][:150]}{"..." if len(msg["text"]) > 150 else ""}</p><div style="display: flex; flex-wrap: wrap; gap: 5px; margin-top: 8px;"><small style="background: #f0f0f0; padding: 2px 6px; border-radius: 10px;">{msg["status"].upper()}</small><small style="background: #e3f2fd; padding: 2px 6px; border-radius: 10px;">{len(msg["models_used"])} models</small>{flagged_html}</div></div>', unsafe_allow_html=True)
                if i >= 4: break
        else: st.info(translate_text("no_conversation", language))
        
        st.subheader(translate_text("session_analytics", language))
        if st.session_state.conversation_history:
            stats = st.session_state.session_stats
            total = stats['total_messages']
            toxic = stats['toxic_messages'] 
            safe = stats['safe_messages']
            fig_donut = go.Figure(data=[go.Pie(labels=['Safe Messages', 'Toxic Messages'], values=[safe, toxic], hole=.3, marker_colors=['#51cf66', '#ff6b6b'])])
            fig_donut.update_layout(title="Message Safety Distribution", height=300, showlegend=True)
            st.plotly_chart(fig_donut, use_container_width=True)
            safety_percentage = (safe / total * 100) if total > 0 else 100
            st.metric("Safety Rate", f"{safety_percentage:.1f}%", f"{'+' if safety_percentage > 80 else ''}{safety_percentage - 80:.1f}%")
        
        st.subheader(translate_text("generate_report", language))
        if st.session_state.conversation_history:
            report_type = st.selectbox("Report Type:", ["Full Session", "Toxic Only", "Summary"])
            if st.button("📊 Generate PDF Report", type="primary"):
                with st.spinner("📄 Generating comprehensive report..."):
                    pdf_buffer = generate_pdf_report(st.session_state.conversation_history, st.session_state.flagged_comments)
                    st.download_button(label="📥 Download Report", data=pdf_buffer, file_name=f"cyberbullying_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf", type="primary")
                    st.success("✅ Report generated successfully!")
            if st.session_state.flagged_comments:
                if st.button("📥 Download Flagged Report"):
                    with st.spinner("📄 Generating flagged report..."):
                        pdf_buffer = generate_flagged_report(st.session_state.flagged_comments)
                        st.download_button(label="📥 Download Flagged Report", data=pdf_buffer, file_name=f"flagged_comments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf", type="primary")
                        st.success("✅ Flagged report generated successfully!")
            st.subheader("💾 Export Options")
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1: 
                if st.button("📊 Export CSV"): 
                    df = pd.DataFrame(st.session_state.conversation_history)
                    st.download_button("📥 Download CSV", df.to_csv(index=False), f"conversation_data_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
            with col_exp2: 
                if st.button("📋 Export JSON"): 
                    import json
                    json_data = json.dumps(st.session_state.conversation_history, indent=2)
                    st.download_button("📥 Download JSON", json_data, f"conversation_data_{datetime.now().strftime('%Y%m%d')}.json", "application/json")
        else: st.info(translate_text("no_data", language))
    
    if st.session_state.feedback:
        st.warning("\n".join(st.session_state.feedback))
    
    if st.session_state.conversation_history:
        st.markdown("---")
        st.header("📈 Advanced Analytics Dashboard")
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Trends", "🎯 Model Performance", "🚩 Flagged Content", "⚡ Real-time Stats"])
        with tab1:
            st.subheader("Toxicity Trends Over Time")
            df_history = pd.DataFrame(st.session_state.conversation_history)
            df_history['datetime'] = pd.to_datetime(f"{datetime.now().strftime('%Y-%m-%d')} " + df_history['timestamp'])
            fig_trend = px.line(df_history, x='datetime', y='toxicity_score', color='status', title="Toxicity Score Timeline", color_discrete_map={'toxic': '#ff6b6b', 'safe': '#51cf66'})
            fig_trend.add_hline(y=toxicity_threshold*100, line_dash="dash", line_color="orange", annotation_text="Safety Threshold")
            st.plotly_chart(fig_trend, use_container_width=True)
            fig_sentiment = px.scatter(df_history, x='datetime', y='sentiment', size='toxicity_score', color='status', title="Sentiment vs Toxicity Analysis", color_discrete_map={'toxic': '#ff6b6b', 'safe': '#51cf66'})
            st.plotly_chart(fig_sentiment, use_container_width=True)
        with tab2:
            st.subheader("AI Model Performance Metrics")
            model_performance = {}
            for msg in st.session_state.conversation_history:
                for model in msg['models_used']:
                    if model not in model_performance:
                        model_performance[model] = {'correct': 0, 'total': 0, 'avg_confidence': []}
                    model_performance[model]['total'] += 1
                    model_performance[model]['avg_confidence'].append(msg['confidence'])
            perf_data = [{'Model': model, 'Total Predictions': stats['total'], 'Average Confidence': np.mean(stats['avg_confidence']), 'Usage %': (stats['total'] / len(st.session_state.conversation_history)) * 100} for model, stats in model_performance.items()]
            perf_df = pd.DataFrame(perf_data)
            st.dataframe(perf_df, use_container_width=True)
            fig_conf = px.box(df_history, x='status', y='confidence', title="Confidence Distribution by Prediction", color='status', color_discrete_map={'toxic': '#ff6b6b', 'safe': '#51cf66'})
            st.plotly_chart(fig_conf, use_container_width=True)
        with tab3:
            st.subheader("Flagged Content Analysis")
            if st.session_state.flagged_comments:
                flagged_df = pd.DataFrame(st.session_state.flagged_comments)
                fig_flagged = px.histogram(flagged_df, x='timestamp', title="Flagged Content Timeline", color_discrete_sequence=['#ff6b6b'])
                st.plotly_chart(fig_flagged, use_container_width=True)
                st.subheader("🚩 Flagged Comments Review")
                for i, flag in enumerate(st.session_state.flagged_comments):
                    with st.expander(f"Flag #{i+1} - {flag['timestamp']} ({flag['toxicity_score']:.1f}%)"):
                        st.write(f"**Content:** {flag['text']}")
                        st.write(f"**Reason:** {flag['reason']}")
                        st.write(f"**Toxicity Score:** {flag['toxicity_score']:.1f}%")
                        col_action1, col_action2, col_action3 = st.columns(3)
                        with col_action1: 
                            if st.button(f"✅ Approve {i}", key=f"approve_{i}"): st.success("Content approved")
                        with col_action2: 
                            if st.button(f"❌ Remove {i}", key=f"remove_{i}"): st.error("Content removed")
                        with col_action3: 
                            if st.button(f"⚠️ Warn User {i}", key=f"warn_{i}"): st.warning("Warning issued")
            else: st.info("No flagged content in this session")
        with tab4:
            st.subheader("Real-time System Statistics")
            col_rt1, col_rt2, col_rt3, col_rt4 = st.columns(4)
            with col_rt1: st.metric("Messages/Hour", len([m for m in st.session_state.conversation_history if datetime.now().hour == datetime.strptime(m['timestamp'], '%H:%M:%S').hour]))
            with col_rt2: 
                recent_toxicity = [m['toxicity_score'] for m in st.session_state.conversation_history[-5:]]
                avg_recent = np.mean(recent_toxicity) if recent_toxicity else 0
                st.metric("Recent Avg Toxicity", f"{avg_recent:.1f}%")
            with col_rt3: st.metric("Active Models", len(selected_models), "Real-time")
            with col_rt4: 
                processing_speed = "Fast" if len(selected_models) <= 2 else "Medium" if len(selected_models) <= 3 else "Comprehensive"
                st.metric("Processing Mode", processing_speed)
            if st.session_state.conversation_history:
                latest_score = st.session_state.conversation_history[-1]['toxicity_score']
                fig_gauge = go.Figure(go.Indicator(mode="gauge+number+delta", value=latest_score, domain={'x': [0, 1], 'y': [0, 1]}, title={'text': "Current Toxicity Level"}, delta={'reference': toxicity_threshold * 100}, gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "darkblue"}, 'steps': [{'range': [0, 30], 'color': "lightgreen"}, {'range': [30, 70], 'color': "yellow"}, {'range': [70, 100], 'color': "red"}], 'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': toxicity_threshold * 100}}))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.markdown("---")
    st.markdown('<div style="text-align: center; color: #666; padding: 20px;"><p>🛡️ <strong>AI-Powered Cyberbullying Detection System</strong></p><p>Powered by XLM-RoBERTa, BiLSTM, SVM & Logistic Regression | Real-time Analysis & Reporting</p><p><em>Protecting digital communities through advanced AI technology</em></p></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()