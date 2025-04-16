import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np

st.set_page_config(page_title="Análise de Sentimento", page_icon="😊", layout="wide")
st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .stTextArea textarea {border-radius: 10px; border: 2px solid #d1d5db; padding: 10px;}
    .stButton>button {background-color: #4b6cb7; color: white; border-radius: 8px; padding: 10px 20px;}
    .stButton>button:hover {background-color: #182848; border-color: #182848;}
    .result-box {background-color: white; border-radius: 10px; padding: 15px; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);}
    .header {color: #3a34eb; font-size: 2.5em; font-weight: bold; text-align: center; margin-bottom: 20px;}
    .subheader {color: #4b6cb7; font-size: 1.5em; font-weight: bold; margin-top: 20px;}
    .info {color: #374151; font-size: 1.1em; text-align: center; margin-bottom: 20px;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">Análise de Sentimento com IA 😊</div>', unsafe_allow_html=True)
st.markdown('<div class="info">Digite frases em inglês separadas por \'.\' (exemplo: I love this. This is bad.) e veja o sentimento de cada uma!</div>', unsafe_allow_html=True)

col_input, col_result = st.columns([1.2, 2])

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
        ax.text(xi, yi + 0.05, f"{label.replace('Positivo ', '').replace('Negativo ', '')}",
                ha='center', fontsize=9, color='green' if label == 'Positivo 😊' else 'red')

    ax.set_ylim(-0.1, 1.5)
    ax.set_title("Tendência de Sentimento ao Longo do Texto")
    ax.set_ylabel("Probabilidade de Sentimento Positivo")
    ax.set_xlabel("Ordem da Frase")
    st.pyplot(fig, use_container_width=True)

results = []

with col_input:
    st.markdown("### Entrada")
    user_input = st.text_area(
        "Digite seu Texto:",
        placeholder="Exemplo: I love my job. But I really hate it when I'm interrupted.",
        height=200
    )
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_button = st.button("Analisar Frases 🚀")

    if analyze_button and user_input:
        with st.spinner("🔍 Validando entrada..."):
            texts = [t.strip() for t in user_input.split(".") if t.strip()]
            time.sleep(0.5)
            if not texts:
                st.error("❌ Nenhum texto válido encontrado.")
                st.stop()

        valid_msg = st.empty()
        valid_msg.success(f"✅ Entrada validada: {len(texts)} frase(s) encontrada(s).")

        tokenizer, model = load_model()
        classes = ["Negativo 😞", "Positivo 😊"]

        for text in texts:
            predicted_label, probabilities = predict(text, tokenizer, model)
            results.append({
                "label": classes[predicted_label],
                "prob": probabilities[0],
                "text": text
            })

        valid_msg.empty()

        st.markdown("""
        <div style='max-height: 400px; overflow-y: auto; padding-right: 10px; margin-top: 15px;'>
        """, unsafe_allow_html=True)

        for result in results:
            label = result["label"]
            prob = result["prob"]
            text = result["text"]
            label_color = "#22c55e" if label == 'Positivo 😊' else "#ef4444"
            with st.container():
                st.markdown(f'<br>', unsafe_allow_html=True)
                st.markdown(f"**Frase**: {text}")
                st.markdown(f"**Rótulo Previsto**: <span style='color:{label_color}'>{label}</span>", unsafe_allow_html=True)
                st.markdown(f"**Probabilidades**: Negativo: {prob[0]:.4f} | Positivo: {prob[1]:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

with col_result:
    if analyze_button and user_input and results:
        st.markdown('<div class="subheader">Gráfico de Mudança de Sentimento</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        plot_sentiment_trend(results)
