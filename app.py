import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np



@st.cache_resource
def load_model():
    with st.spinner("🔄 Carregando modelo de IA..."):
        tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")
        model = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english")
        time.sleep(1)
    success_box = st.empty()
    success_box.success("✅ Modelo carregado com sucesso!")
    time.sleep(1.5)
    success_box.empty()
    return tokenizer, model

def predict(text, tokenizer, model):
    with st.spinner(f"🧠 Analisando: {text}..."):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_label = torch.argmax(probabilities, dim=1).item()
        time.sleep(0.5)
    return predicted_label, probabilities

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import time

# CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Análise de Sentimento", page_icon="😊", layout="wide")
st.markdown("""
    <style>
    .main {background-color: #f8fafc; padding: 10px;}
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #cbd5e1;
        padding: 12px;
        font-size: 14px;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        transition: 0.2s ease-in-out;
    }
    .header {
        color: #1f2937;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 10px;
    }
    .info {
        color: #4b5563;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 25px;
    }
    .result-box {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .subheader {
        font-size: 1.4rem;
        color: #1e40af;
        margin: 10px 0;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# HEADER
st.markdown('<div class="header">Análise de Sentimento com IA 😊</div>', unsafe_allow_html=True)
st.markdown('<div class="info">Digite frases em inglês separadas por "." e veja a análise individual de sentimento.</div>', unsafe_allow_html=True)

# LAYOUT
col_input, col_result = st.columns([1.2, 2])

# PLOT
def plot_sentiment_trend(results):
    x = np.arange(1, len(results) + 1)
    y = np.array([r["prob"][1].item() for r in results])
    labels = [r["label"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 4))

    if len(results) >= 4:
        x_smooth = np.linspace(x.min(), x.max(), 300)
        spline = make_interp_spline(x, y, k=3)
        y_smooth = spline(x_smooth)
        ax.plot(x_smooth, y_smooth, color='blue', linewidth=2)
    else:
        ax.plot(x, y, marker='o', linestyle='-', color='blue', linewidth=2)

    ax.scatter(x, y, color='blue')

    for i, (xi, yi, label) in enumerate(zip(x, y, labels)):
        ax.text(xi, yi + 0.05, label.replace("Positivo ", "").replace("Negativo ", ""),
                ha='center', fontsize=9, color='green' if label == 'Positivo 😊' else 'red')

    ax.set_ylim(-0.2, 1.5)
    ax.set_title("Tendência de Sentimento")
    ax.set_ylabel("Prob. Positivo")
    ax.set_xlabel("Frase")
    st.pyplot(fig, use_container_width=True)

# VARIÁVEIS
results = []

# COLUNA DE ENTRADA
with col_input:
    st.markdown("### Texto de Entrada")
    user_input = st.text_area("Digite seu Texto:",
        placeholder="Exemplo: I love this. This is bad.",
        height=200
    )
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_button = st.button("Analisar Sentimentos 🚀")

    if analyze_button and user_input:
        with st.spinner("🔍 Analisando..."):
            texts = [t.strip() for t in user_input.split(".") if t.strip()]
            time.sleep(0.5)
            if not texts:
                st.error("❌ Nenhuma frase válida encontrada.")
                st.stop()

        valid_msg = st.empty()
        valid_msg.success(f"✅ {len(texts)} frase(s) detectada(s).")

        # Exemplo fictício para testes
        tokenizer, model = load_model()  # sua função de carregamento
        classes = ["Negativo 😞", "Positivo 😊"]

        for text in texts:
            predicted_label, probabilities = predict(text, tokenizer, model)
            results.append({
                "label": classes[predicted_label],
                "prob": probabilities[0],
                "text": text
            })

        valid_msg.empty()

        st.markdown("<div style='max-height: 400px; overflow-y: auto;'>", unsafe_allow_html=True)

        for result in results:
            label = result["label"]
            prob = result["prob"]
            text = result["text"]
            label_color = "#22c55e" if label == 'Positivo 😊' else "#ef4444"
            st.markdown(f'''
                <div class="result-box">
                    <b>Frase:</b> <span style="color:#1e293b;">"{text}"</span><br>
                    <b>Sentimento:</b> <span style="color:{label_color}">{label}</span><br>
                    <b>Probabilidades:</b> Negativo: {prob[0]:.4f} | Positivo: {prob[1]:.4f}
                </div>
            ''', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# COLUNA DO GRÁFICO
with col_result:
    if analyze_button and user_input and results:
        st.markdown('<div class="subheader">Visualização Gráfica</div>', unsafe_allow_html=True)
        plot_sentiment_trend(results)
