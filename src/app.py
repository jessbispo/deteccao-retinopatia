"""
app.py
Interface web para classificação de retinopatia diabética.

Uso:
    streamlit run src/app.py
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st
from PIL import Image

from preprocessing import preprocess_retina_image
from model import load_model
from gradcam import GradCAM
import sys


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Detecção de Retinopatia Diabética", page_icon="", layout="wide"
)

st.title("Classificação de Retinopatia Diabética")
st.markdown("""
**Computação Visual - Universidade Presbiteriana Mackenzie**
Bruna Aguiar - Gabriel Geronazzo - Jessica Bispo - Lucas Sarai
""")
st.divider()


# ---------------------------------------------------------------------------
# Carregamento do modelo (cache para não recarregar a cada interação)
# ---------------------------------------------------------------------------


@st.cache_resource
def get_model():
    """Carrega o modelo uma única vez e mantém em cache."""
    checkpoint = resource_path("modelo_retina.pth")
    model, device = load_model(
        checkpoint_path=checkpoint if os.path.exists(checkpoint) else None
    )
    return model, device


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configurações")

    st.subheader("Pré-processamento")
    use_clahe = st.toggle(
        "CLAHE (melhoria de contraste)",
        value=True,
        help="Equaliza o histograma localmente para realçar lesões",
    )
    use_gaussian = st.toggle(
        "Subtração Gaussiana de Fundo",
        value=True,
        help="Remove iluminação não uniforme entre imagens",
    )
    gaussian_strength = st.slider(
        "Intensidade da correção",
        0.0,
        1.0,
        0.3,
        0.05,
        help="0 = sem efeito | 1 = Ben Graham puro (pode gerar artefatos)",
        disabled=not use_gaussian,
    )
    use_circular = st.toggle(
        "Máscara Circular",
        value=True,
        help="Foca o modelo na região circular da retina",
    )
    use_green = st.toggle(
        "Canal Verde (alternativo)",
        value=False,
        help="Realça vasos sanguíneos via canal G",
    )
    show_stages = st.toggle("Mostrar etapas do processamento", value=True)

    st.subheader("Classificação")
    threshold = st.slider(
        "Limiar de decisão",
        0.1,
        0.9,
        0.5,
        0.05,
        help="Probabilidade mínima para classificar como retinopatia",
    )

    st.divider()
    st.info(
        "Esta ferramenta é para fins acadêmicos e não substitui avaliação médica profissional."
    )


# ---------------------------------------------------------------------------
# Upload de imagem
# ---------------------------------------------------------------------------

col_upload, col_info = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Envie uma imagem de retina", type=["jpg", "jpeg", "png", "tiff", "bmp"]
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


# ---------------------------------------------------------------------------
# Processamento e exibição dos resultados
# ---------------------------------------------------------------------------

if uploaded_file is not None:
    img_pil = Image.open(uploaded_file).convert("RGB")

    st.divider()
    st.header("Análise da Imagem")

    # --- Pré-processamento ---
    with st.spinner("Aplicando pré-processamento..."):
        tensor, stages = preprocess_retina_image(
            img_pil,
            use_clahe=use_clahe,
            use_gaussian=use_gaussian,
            gaussian_strength=gaussian_strength,
            use_circular_mask=use_circular,
            use_green_channel=use_green,
            return_stages=True,
        )

    if show_stages:
        st.subheader("Etapas do Pré-processamento")
        st.markdown("*Cada etapa aplica uma técnica de processamento de imagem*")

        stage_labels = {
            "0_original": "Original",
            "1_sem_borda": "Sem Borda Preta (Thresholding + BBox)",
            "2_resize": "Redimensionamento (Letterboxing 224x224)",
            "3_mascara_circular": "Máscara Circular (ROI)",
            "4a_clahe": "CLAHE (Equalização Adaptativa)",
            "4b_gaussian": "Subtração Gaussiana (Remoção de Fundo)",
            "4_canal_verde": "Canal Verde (Realce Vascular)",
            "5_final_rgb": "Imagem Final (Pronta para o Modelo)",
        }

        stage_keys = [k for k in stage_labels if k in stages]
        cols = st.columns(min(len(stage_keys), 4))

        for i, key in enumerate(stage_keys):
            with cols[i % 4]:
                st.image(stages[key], caption=stage_labels[key], width="stretch")

    # --- Classificação ---
    st.subheader("Classificação")

    with st.spinner("Classificando..."):
        try:
            model, device = get_model()
            result = model.predict(tensor.to(device), threshold=threshold)
        except Exception as e:
            st.error(f"Erro no modelo: {e}")
            st.stop()

    col_result, col_prob = st.columns(2)

    with col_result:
        if result["has_dr"]:
            st.error(f"## {result['label']}")
        else:
            st.success(f"## {result['label']}")

    with col_prob:
        st.metric("Probabilidade de Retinopatia", f"{result['probability'] * 100:.1f}%")
        st.metric("Confiança da predição", f"{result['confidence'] * 100:.1f}%")
        st.progress(result["probability"])

    # --- Grad-CAM ---
    st.subheader("Visualização Grad-CAM")
    st.markdown("""
    O **Grad-CAM** mostra quais regiões da retina influenciaram mais a decisão do modelo.
    - Azul/Frio: baixa influência
    - Vermelho/Quente: alta influência (regiões determinantes)
    """)

    with st.spinner("Gerando heatmap Grad-CAM..."):
        try:
            gradcam = GradCAM(model)
            final_img_rgb = stages.get(
                "5_final_rgb",
                stages.get("4b_gaussian", stages.get("4a_clahe", stages["2_resize"])),
            )
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
            st.image(final_img_rgb, caption="Imagem Pré-processada", width="stretch")
        with col2:
            st.image(viz["heatmap"], caption="Heatmap Grad-CAM", width="stretch")
        with col3:
            st.image(viz["overlay"], caption="Sobreposição (Overlay)", width="stretch")

        # Barra de referencia do colormap
        fig, ax = plt.subplots(figsize=(6, 0.5))
        fig.patch.set_alpha(0)
        norm = mcolors.Normalize(vmin=0, vmax=1)
        cb = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.jet),
            cax=ax,
            orientation="horizontal",
        )
        cb.set_label("Nível de ativação (baixo -> alto)", color="white")
        cb.ax.xaxis.set_tick_params(color="white", labelcolor="white")
        st.pyplot(fig, width="stretch")

    # --- Detalhes técnicos ---
    with st.expander("Detalhes Técnicos"):
        st.markdown(f"""
        **Modelo:** ResNet18 com fine-tuning
        **Dispositivo:** `{device}`
        **Tensor de entrada:** `{tuple(tensor.shape)}`
        **Probabilidade bruta:** `{result["probability"]:.6f}`
        **Limiar utilizado:** `{threshold}`

        **Técnicas de CV aplicadas:**
        - {"V" if use_clahe else "X"} CLAHE
        - {"V" if use_gaussian else "X"} Subtração Gaussiana de Fundo
        - {"V" if use_circular else "X"} Máscara Circular
        - {"V" if use_green else "X"} Canal Verde
        """)

else:
    st.info("Faça upload de uma imagem de retina para começar a análise.")

    st.subheader("Pipeline do Sistema")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**1. Entrada**\n\nUpload da imagem de retina (fundoscopia)")
    with col2:
        st.markdown(
            "**2. Pré-processamento**\n\nCLAHE, Gaussiana, Máscara Circular, Resize"
        )
    with col3:
        st.markdown("**3. Classificação**\n\nResNet18 -> Normal ou Retinopatia")
    with col4:
        st.markdown(
            "**4. Explicabilidade**\n\nGrad-CAM -> Heatmap das regiões ativadas"
        )
