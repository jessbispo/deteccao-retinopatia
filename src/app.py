"""
app.py
======
Interface web com Streamlit para classificação de retinopatia diabética.

Uso:
    streamlit run app.py

Funcionalidades:
- Upload de imagem de retina
- Visualização de cada etapa do pré-processamento
- Classificação com probabilidade
- Heatmap Grad-CAM sobreposto
"""

import streamlit as st
import numpy as np
from PIL import Image
import cv2
import torch
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from preprocessing import preprocess_retina_image
from model import load_model
from gradcam import GradCAM
import sys

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# ──────────────────────────────────────────────
# Configuração da página
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Detecção de Retinopatia Diabética",
    page_icon="",
    layout="wide"
)

st.title("Classificação de Retinopatia Diabética")
st.markdown("""
**Computação Visual — Universidade Presbiteriana Mackenzie**  
Bruna Aguiar · Gabriel Geronazzo · Jessica Bispo · Lucas Sarai
""")
st.divider()


# ──────────────────────────────────────────────
# Carregamento do modelo (cache para não recarregar)
# ──────────────────────────────────────────────

@st.cache_resource
def get_model():
    """Carrega o modelo uma única vez e mantém em cache."""
    checkpoint = resource_path("modelo_retina.pth")
    model, device = load_model(
        checkpoint_path=checkpoint if os.path.exists(checkpoint) else None
    )
    return model, device


# ──────────────────────────────────────────────
# Sidebar — Configurações
# ──────────────────────────────────────────────

with st.sidebar:
    st.header("Configurações")
    
    st.subheader("Pré-processamento")
    use_clahe = st.toggle("CLAHE (melhoria de contraste)", value=True,
                           help="Equalização adaptativa de histograma por regiões")
    use_gaussian = st.toggle("Subtração Gaussiana de Fundo", value=True,
                              help="Remove iluminação não uniforme (Ben Graham's method)")
    use_circular = st.toggle("Máscara Circular", value=True,
                              help="Remove cantos irrelevantes da imagem")
    use_green = st.toggle("Canal Verde (alternativo)", value=False,
                           help="Realça vasos sanguíneos via canal G")
    
    show_stages = st.toggle("Mostrar etapas do processamento", value=True)
    
    st.subheader("Classificação")
    threshold = 0.7 #st.slider("Limiar de decisão", 0.1, 0.9, 0.5, 0.05,
                           #help="Probabilidade mínima para classificar como retinopatia") 
    
    st.divider()
    st.info("**Atenção:** Esta ferramenta é para fins acadêmicos e não substitui avaliação médica profissional.")


# ──────────────────────────────────────────────
# Upload de imagem
# ──────────────────────────────────────────────

col_upload, col_info = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "📤 Envie uma imagem de retina",
        type=["jpg", "jpeg", "png", "tiff", "bmp"],
        help="Formatos suportados: JPG, PNG, TIFF, BMP"
    )

with col_info:
    st.markdown("""
    **Como usar:**
    1. Envie uma imagem de retina (fundoscopia)
    2. Configure o pré-processamento na barra lateral
    3. Veja o resultado e o heatmap Grad-CAM
    
    **Sobre o modelo:**
    - Arquitetura: ResNet18
    - Classificação: Normal / Retinopatia
    """)


# ──────────────────────────────────────────────
# Processamento e exibição
# ──────────────────────────────────────────────

if uploaded_file is not None:
    
    # Carrega a imagem
    img_pil = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img_pil)
    
    st.divider()
    st.header("Análise da Imagem")
    
    # ── Etapas de pré-processamento ────────────────────────────
    with st.spinner("Aplicando pré-processamento..."):
        tensor, stages = preprocess_retina_image(
            img_pil,
            use_clahe=use_clahe,
            use_gaussian=use_gaussian,
            use_circular_mask=use_circular,
            use_green_channel=use_green,
            return_stages=True
        )
    
    if show_stages:
        st.subheader("Etapas do Pré-Processamento")
        st.markdown("*Cada etapa aplica uma técnica de processamento de imagem*")
        
        stage_labels = {
            "0_original": "Original",
            "1_sem_borda": "Sem Borda Preta\n(Thresholding + BBox)",
            "2_resize": "Redimensionamento\n(Letterboxing 224×224)",
            "3_mascara_circular": "Máscara Circular\n(ROI Circular)",
            "4a_clahe": "CLAHE\n(Equalização Adaptativa)",
            "4b_gaussian": "Subtração Gaussiana\n(Remoção de Fundo)",
            "4_canal_verde": "Canal Verde\n(Realce Vascular)",
            "5_final_rgb": "Imagem Final\n(Pronta para o Modelo)"
        }
        
        stage_keys = [k for k in stage_labels.keys() if k in stages]
        cols = st.columns(min(len(stage_keys), 4))
        
        for i, key in enumerate(stage_keys):
            with cols[i % 4]:
                st.image(stages[key], caption=stage_labels[key], use_column_width=True)
    
    # ── Classificação ──────────────────────────────────────────
    st.subheader("Classificação")
    
    with st.spinner("Classificando..."):
        try:
            model, device = get_model()
            result = model.predict(tensor.to(device), threshold=threshold)
        except Exception as e:
            st.error(f"Erro no modelo: {e}")
            st.stop()
    
    # Exibe resultado com destaque visual
    col_result, col_prob = st.columns(2)
    
    with col_result:
        if result["has_dr"]:
            st.error(f"## {result['label']}")
        else:
            st.success(f"## {result['label']}")
    
    with col_prob:
        prob_pct = result['probability'] * 100
        confidence_pct = result['confidence'] * 100
        
        st.metric("Probabilidade de Retinopatia", f"{prob_pct:.1f}%")
        st.metric("Confiança da predição", f"{confidence_pct:.1f}%")
        
        # Barra de progresso
        st.progress(result['probability'])
    
    # ── Grad-CAM ───────────────────────────────────────────────
    st.subheader("Visualização Grad-CAM")
    st.markdown("""
    O **Grad-CAM** mostra quais regiões da retina influenciaram mais a decisão do modelo.
    - **Azul/Frio**: baixa influência
    - **Vermelho/Quente**: alta influência (regiões determinantes)
    """)
    
    with st.spinner("Gerando heatmap Grad-CAM..."):
        try:
            gradcam = GradCAM(model)
            
            # Imagem pré-processada para sobrepor o heatmap
            final_img_rgb = stages.get("5_final_rgb", stages.get("4b_gaussian",
                           stages.get("4a_clahe", stages["2_resize"])))
            
            viz = gradcam.generate_full_visualization(
                tensor, final_img_rgb, device=device
            )
            gradcam.remove_hooks()
        except Exception as e:
            st.error(f"Erro no Grad-CAM: {e}")
            viz = None
    
    if viz is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(final_img_rgb, caption="Imagem Pré-processada", use_column_width=True)
        
        with col2:
            st.image(viz["heatmap"], caption="Heatmap Grad-CAM", use_column_width=True)
        
        with col3:
            st.image(viz["overlay"], caption="Sobreposição (Overlay)", use_column_width=True)
        
        # Colorbar para referência
        fig, ax = plt.subplots(figsize=(6, 0.5))
        fig.patch.set_alpha(0)
        cmap = plt.cm.jet
        norm = mcolors.Normalize(vmin=0, vmax=1)
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                          cax=ax, orientation='horizontal')
        cb.set_label('Nível de ativação (baixo → alto)', color='white')
        cb.ax.xaxis.set_tick_params(color='white', labelcolor='white')
        st.pyplot(fig, use_container_width=True)
    
    # ── Informações técnicas ───────────────────────────────────
    with st.expander("Detalhes Técnicos"):
        st.markdown(f"""
        **Modelo:** ResNet18 com fine-tuning  
        **Dispositivo:** `{device}`  
        **Tensor de entrada:** `{tuple(tensor.shape)}`  
        **Probabilidade bruta:** `{result['probability']:.6f}`  
        **Limiar utilizado:** `{threshold}`
        
        **Técnicas de CV aplicadas:**
        - {'Sim' if use_clahe else 'Não'} CLAHE (Contrast Limited Adaptive Histogram Equalization)
        - {'Sim' if use_gaussian else 'Não'} Gaussian Background Subtraction
        - {'Sim' if use_circular else 'Não'} Circular Masking (ROI)
        - {'Sim' if use_green else 'Não'} Green Channel Extraction
        """)

else:
    # Estado inicial sem imagem
    st.info("Faça upload de uma imagem de retina para começar a análise.")
    
    # Exemplo visual do pipeline
    st.subheader("Pipeline do Sistema")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        **1. Entrada**
        
        Upload da imagem de retina (fundoscopia)
        """)
    with col2:
        st.markdown("""
        **2. Pré-processamento**
        
        CLAHE, Gaussian, Máscara Circular, Resize
        """)
    with col3:
        st.markdown("""
        **3. Classificação**
        
        ResNet18 → Normal ou Retinopatia
        """)
    with col4:
        st.markdown("""
        **4. Explicabilidade**
        
        Grad-CAM → Heatmap das regiões ativadas
        """)
