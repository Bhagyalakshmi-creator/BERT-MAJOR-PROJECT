import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import warnings
import importlib.metadata

warnings.filterwarnings('ignore')

# -------------------------------
# Page config & CSS
# -------------------------------
st.set_page_config(
    page_title="Cyberbullying Detection System",
    page_icon="🚫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ultra-compact CSS for single-page view
st.markdown("""
<style>
    .main-header {font-size: 1.6rem; color: #FF4B4B; text-align: center; margin-bottom: 0.3rem; font-weight: bold;}
    .prediction-box {background-color: #f8f9fa; padding: 0.4rem; border-radius: 4px; margin: 0.3rem 0;
                     box-shadow: 0 1px 2px rgba(0,0,0,0.05); text-align: center;}
    .abusive {color: #FF4B4B; font-weight: bold; font-size: 1.1rem; margin: 0.1rem 0;}
    .non-abusive {color: #00CC96; font-weight: bold; font-size: 1.1rem; margin: 0.1rem 0;}
    .metric-box {background-color: white; padding: 0.3rem; border-radius: 3px; box-shadow: 0 1px 2px rgba(0,0,0,0.04);
                 margin: 0.2rem 0; text-align: center; border-left: 2px solid;}
    .footer {text-align: center; margin-top: 0.6rem; padding: 0.4rem; color: #6c757d; font-size: 0.7rem;}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Model definition
# -------------------------------
class BERTClassifier(nn.Module):
    def __init__(self, num_labels):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

class AbuseDetector:
    def __init__(self, model_path, label_names):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_names = label_names
        self.model = BERTClassifier(len(label_names))
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights '{model_path}' not found!")
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    def predict(self, text):
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
        return probabilities

# -------------------------------
# Labels & Thresholds
# -------------------------------
LABEL_NAMES = ["Ethnicity", "Other", "Political", "Racism", "Religion", "Sexual / Gender", "Threat", "Troll"]
# DEFAULT_THRESHOLDS = {"Troll": 0.85, "Sexual / Gender": 0.75, "Threat": 0.70,
#                       "Other": 0.60, "Ethnicity": 0.60, "Racism": 0.60, "Religion": 0.65, "Political": 0.70}
DEFAULT_THRESHOLDS= {"Threat": 0.6,
    "Racism": 0.4,
    "Religion": 0.4,
    "Sexual / Gender": 0.35,
    "Troll": 0.35,
    "Political": 0.4,
    "Ethnicity": 0.4,
    "Other": 0.45}
EXAMPLE_TEXTS = [
    "You're so stupid and worthless, nobody likes you",
    "I respect everyone's beliefs and opinions",
    "That was an amazing performance, great job!",
    "People like you should go back to where you came from, you are worthless",
    "Your beliefs don’t excuse bad behavior."
]

# -------------------------------
# Main Streamlit app
# -------------------------------
def main():
    if 'thresholds' not in st.session_state:
        st.session_state.thresholds = DEFAULT_THRESHOLDS.copy()
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.info("bert_cyberbullying.pt", icon="✅")
        for label in LABEL_NAMES:
            st.session_state.thresholds[label] = st.slider(
                f"{label}", 0.0, 1.0, DEFAULT_THRESHOLDS[label], 0.05, key=f"threshold_{label}")
        st.markdown("---")
        st.markdown("**Example Texts**")
        for i, text in enumerate(EXAMPLE_TEXTS):
            if st.button(text, key=f"example_{i}", use_container_width=True):
                st.session_state.example_text = text
    st.markdown('<h1 class="main-header">Cyberbullying Detection System</h1>', unsafe_allow_html=True)
    default_text = st.session_state.get("example_text", "")
    text_input = st.text_area("Type or paste text here:", height=70,
                              placeholder="Enter text to check for cyberbullying...",
                              value=default_text, label_visibility="collapsed")
    analyze_btn = st.button("Analyze Text", disabled=not text_input.strip(), use_container_width=True)
    if 'detector' not in st.session_state:
        try:
            with st.spinner("Loading model..."):
                st.session_state.detector = AbuseDetector("bert_cyberbullying.pt", LABEL_NAMES)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()
    if analyze_btn and text_input.strip():
        with st.spinner("Analyzing..."):
            probs = st.session_state.detector.predict(text_input)
            results = dict(zip(LABEL_NAMES, probs))
            abusive_labels = [lbl for lbl, prob in results.items() if prob >= st.session_state.thresholds[lbl]]
            is_abusive = len(abusive_labels) > 0
            max_toxic_prob = max(probs) if probs.any() else 0
            clean_prob = 1 - max_toxic_prob
            if is_abusive:
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("**Detection Result**")
                    st.markdown(f"""
                    <div class="prediction-box">
                        <div class="abusive">🚫 ABUSIVE</div>
                        <div style="font-size: 0.8rem;">Detected: {', '.join(abusive_labels)}</div>
                        <div style="font-size: 0.75rem; color:#d32f2f;">Toxic: {max_toxic_prob:.1%}</div>
                        <div style="font-size: 0.75rem; color:#2e7d32;">Clean: {clean_prob:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number", value=max_toxic_prob*100,
                        domain={'x':[0,1],'y':[0,1]}, title={'text':"Toxicity Meter",'font':{'size':12}},
                        gauge={'axis':{'range':[0,100]},'bar':{'color':"darkred",'thickness':0.15},
                               'steps':[{'range':[0,30],'color':"lightgreen"},
                                        {'range':[30,70],'color':"yellow"},
                                        {'range':[70,100],'color':"red"}]}))
                    fig_gauge.update_layout(height=180, margin=dict(l=5, r=5, t=25, b=5))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                with col2:
                    st.markdown("**Category Analysis**")
                    cols = st.columns(2)
                    for i,(label,prob) in enumerate(results.items()):
                        color = "#FF4B4B" if prob >= st.session_state.thresholds[label] else "#00CC96"
                        emoji = "⚠️" if prob >= st.session_state.thresholds[label] else "✅"
                        with cols[i % 2]:
                            st.markdown(f"""
                            <div class="metric-box" style="border-left-color: {color};">
                                <div style="font-size: 0.8rem;">{emoji} {label}</div>
                                <div style="font-size: 0.9rem; color: {color};">{prob:.1%}</div>
                                <div style="font-size: 0.65rem; color: #6c757d;">Thresh: {st.session_state.thresholds[label]:.0%}</div>
                            </div>
                            """, unsafe_allow_html=True)
                # Show About section only for abusive text
                st.markdown("---")
                st.markdown("**About**")
                about_col1, about_col2 = st.columns([2,1])
                with about_col1:
                    st.markdown("""
                    <div style="font-size:0.9rem; line-height:1.2;">
                    • <b>Ethnicity</b>: Ethnic background attacks<br>
                    • <b>Political</b>: Political belief targeting<br>
                    • <b>Racism</b>: Racist discrimination<br>
                    • <b>Religion</b>: Religious belief attacks<br>
                    • <b>Sexual / Gender</b>: Gender/sexual harassment<br>
                    • <b>Threat</b>: Direct/indirect threats<br>
                    • <b>Troll</b>: Provocative content<br>
                    • <b>Other</b>: Miscellaneous or might have sarcasm
                    </div>
                    """, unsafe_allow_html=True)
                with about_col2:
                    st.markdown("**System Info**")
                    st.write(f"Py: {sys.version.split()[0]}")
                    st.write(f"Torch: {torch.__version__}")
                    st.write(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
            else:
                # Only clean message, no categories or about section
                st.markdown(f"""
                <div class="prediction-box">
                    <div class="non-abusive">✅ CLEAN</div>
                    <div style="font-size: 0.8rem;">No cyberbullying detected</div>
                </div>
                """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""<div class="footer">Built with 🤗 Transformers and Streamlit</div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
