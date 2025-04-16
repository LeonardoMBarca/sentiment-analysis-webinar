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

st.set_page_config(page_title="Análise de Sentimento", page_icon="😊", layout="wide")

# ESTILO FIXO E MINIMALISTA
st.markdown("""
    <style>
    html, body, .main {
        background-color: #ffffff;
        overflow: hidden !important;
    }
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #d1d5db;
        padding: 10px;
        font-size: 14px;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #2563eb;
    }
    .header {
        color: #1f2937;
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }
    .info {
        color: #4b5563;
        font-size: 1.05rem;
        text-align: center;
        margin-bottom: 20px;
    }
    .result-box {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        font-size: 14px;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .subheader {
        font-size: 1.3rem;
        color: #1e40af;
        margin-bottom: 10px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# TÍTULO E INSTRUÇÕES
st.markdown('<div class="header">Análise de Sentimento com IA 😊</div>', unsafe_allow_html=True)
st.markdown('<div class="info">Digite frases em inglês separadas por "." para ver a análise emocional de cada uma.</div>', unsafe_allow_html=True)

# LAYOUT TRIPARTIDO: ENTRADA | RESULTADOS | GRÁFICO
col_input, col_result, col_plot = st.columns([1, 1, 1])

# FUNÇÃO DE GRÁFICO
def plot_sentiment_trend(results):
    x = np.arange(1, len(results) + 1)
    y = np.array([r["prob"][1].item() for r in results])
    labels = [r["label"] for r in results]

    fig, ax = plt.subplots(figsize=(4, 3))

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

    ax.set_ylim(-0.1, 1.5)
    ax.set_title("Tendência")
    ax.set_ylabel("Positivo")
    ax.set_xlabel("Frase")
    ax.set_xticks(x)
    st.pyplot(fig, use_container_width=True)

# VARIÁVEIS
results = []

# ENTRADA
with col_input:
    st.markdown("### Entrada")
    user_input = st.text_area("Texto:", placeholder="Ex: I love it. I hate noise.", height=200)
    analyze_button = st.button("Analisar 🚀")

# RESULTADOS
with col_result:
    if analyze_button and user_input:
        texts = [t.strip() for t in user_input.split(".") if t.strip()]
        time.sleep(0.3)
        if not texts:
            st.error("❌ Nenhuma frase válida encontrada.")
            st.stop()

        tokenizer, model = load_model()
        classes = ["Negativo 😞", "Positivo 😊"]

        for text in texts:
            predicted_label, probabilities = predict(text, tokenizer, model)
            results.append({
                "label": classes[predicted_label],
                "prob": probabilities[0],
                "text": text
            })

        st.markdown('<div class="subheader">Resultados</div>', unsafe_allow_html=True)

        for result in results:
            label = result["label"]
            prob = result["prob"]
            text = result["text"]
            label_color = "#22c55e" if label == 'Positivo 😊' else "#ef4444"

            st.markdown(f'''
            <div class="result-box">
                <div><b>Frase:</b> {text}</div>
                <div><b>Sentimento:</b> <span style="color:{label_color}">{label}</span></div>
                <div><b>Prob.:</b> Neg: {prob[0]:.2f} | Pos: {prob[1]:.2f}</div>
            </div>
            ''', unsafe_allow_html=True)

# GRÁFICO
with col_plot:
    if results:
        st.markdown('<div class="subheader">Visualização</div>', unsafe_allow_html=True)
        plot_sentiment_trend(results)
