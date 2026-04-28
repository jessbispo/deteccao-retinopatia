"""
app.py
Interface web para classificacao de retinopatia diabetica.

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


# ---------------------------------------------------------------------------
# Configuracao da pagina
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Deteccao de Retinopatia Diabetica",
    page_icon="",
    layout="wide"
)

st.title("Classificacao de Retinopatia Diabetica")
st.markdown("""
**Computacao Visual - Universidade Presbiteriana Mackenzie**
Bruna Aguiar - Gabriel Geronazzo - Jessica Bispo - Lucas Sarai
""")
st.divider()


# ---------------------------------------------------------------------------
# Carregamento do modelo (cache para nao recarregar a cada interacao)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_model():
    checkpoint = "modelo_retina.pth"
    return load_model(
        checkpoint_path=checkpoint if os.path.exists(checkpoint) else None
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuracoes")

    st.subheader("Pre-processamento")
    use_clahe = st.toggle("CLAHE (melhoria de contraste)", value=True,
                           help="Equaliza o histograma localmente para realcar lesoes")
    use_gaussian = st.toggle("Subtracao Gaussiana de Fundo", value=True,
                              help="Remove iluminacao nao uniforme entre imagens")
    gaussian_strength = st.slider("Intensidade da correcao", 0.0, 1.0, 0.3, 0.05,
                                   help="0 = sem efeito | 1 = Ben Graham puro (pode gerar artefatos)",
                                   disabled=not use_gaussian)
    use_circular = st.toggle("Mascara Circular", value=True,
                              help="Foca o modelo na regiao circular da retina")
    use_green = st.toggle("Canal Verde (alternativo)", value=False,
                           help="Realca vasos sanguineos via canal G")
    show_stages = st.toggle("Mostrar etapas do processamento", value=True)

    st.subheader("Classificacao")
    threshold = st.slider("Limiar de decisao", 0.1, 0.9, 0.5, 0.05,
                           help="Probabilidade minima para classificar como retinopatia")

    st.divider()
    st.info("Esta ferramenta e para fins academicos e nao substitui avaliacao medica profissional.")


# ---------------------------------------------------------------------------
# Upload de imagem
# ---------------------------------------------------------------------------

col_upload, col_info = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Envie uma imagem de retina",
        type=["jpg", "jpeg", "png", "tiff", "bmp"]
    )

with col_info:
    st.markdown("""
    **Como usar:**
    1. Envie uma imagem de retina (fundoscopia)
    2. Configure o pre-processamento na barra lateral
    3. Veja o resultado e o heatmap Grad-CAM

    **Sobre o modelo:**
    - Arquitetura: ResNet18
    - Classificacao: Normal / Retinopatia
    """)


# ---------------------------------------------------------------------------
# Processamento e exibicao dos resultados
# ---------------------------------------------------------------------------

if uploaded_file is not None:

    img_pil = Image.open(uploaded_file).convert("RGB")

    st.divider()
    st.header("Analise da Imagem")

    # --- Pre-processamento ---
    with st.spinner("Aplicando pre-processamento..."):
        tensor, stages = preprocess_retina_image(
            img_pil,
            use_clahe=use_clahe,
            use_gaussian=use_gaussian,
            gaussian_strength=gaussian_strength,
            use_circular_mask=use_circular,
            use_green_channel=use_green,
            return_stages=True
        )

    if show_stages:
        st.subheader("Etapas do Pre-processamento")
        st.markdown("*Cada etapa aplica uma tecnica de processamento de imagem*")

        stage_labels = {
            "0_original":          "Original",
            "1_sem_borda":         "Sem Borda Preta (Thresholding + BBox)",
            "2_resize":            "Redimensionamento (Letterboxing 224x224)",
            "3_mascara_circular":  "Mascara Circular (ROI)",
            "4a_clahe":            "CLAHE (Equalizacao Adaptativa)",
            "4b_gaussian":         "Subtracao Gaussiana (Remocao de Fundo)",
            "4_canal_verde":       "Canal Verde (Realce Vascular)",
            "5_final_rgb":         "Imagem Final (Pronta para o Modelo)"
        }

        stage_keys = [k for k in stage_labels if k in stages]
        cols = st.columns(min(len(stage_keys), 4))

        for i, key in enumerate(stage_keys):
            with cols[i % 4]:
                st.image(stages[key], caption=stage_labels[key], width="stretch")

    # --- Classificacao ---
    st.subheader("Classificacao")

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
        st.metric("Probabilidade de Retinopatia", f"{result['probability']*100:.1f}%")
        st.metric("Confianca da predicao",        f"{result['confidence']*100:.1f}%")
        st.progress(result['probability'])

    # --- Grad-CAM ---
    st.subheader("Visualizacao Grad-CAM")
    st.markdown("""
    O **Grad-CAM** mostra quais regioes da retina influenciaram mais a decisao do modelo.
    - Azul/Frio: baixa influencia
    - Vermelho/Quente: alta influencia (regioes determinantes)
    """)

    with st.spinner("Gerando heatmap Grad-CAM..."):
        try:
            gradcam = GradCAM(model)
            final_img_rgb = stages.get("5_final_rgb",
                            stages.get("4b_gaussian",
                            stages.get("4a_clahe", stages["2_resize"])))
            viz = gradcam.generate_full_visualization(tensor, final_img_rgb,
                                                      device=device)
            gradcam.remove_hooks()
        except Exception as e:
            st.error(f"Erro no Grad-CAM: {e}")
            viz = None

    if viz is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(final_img_rgb,   caption="Imagem Pre-processada", width="stretch")
        with col2:
            st.image(viz["heatmap"],  caption="Heatmap Grad-CAM",      width="stretch")
        with col3:
            st.image(viz["overlay"],  caption="Sobreposicao (Overlay)", width="stretch")

        # Barra de referencia do colormap
        fig, ax = plt.subplots(figsize=(6, 0.5))
        fig.patch.set_alpha(0)
        norm = mcolors.Normalize(vmin=0, vmax=1)
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.jet),
                          cax=ax, orientation='horizontal')
        cb.set_label('Nivel de ativacao (baixo -> alto)', color='white')
        cb.ax.xaxis.set_tick_params(color='white', labelcolor='white')
        st.pyplot(fig, width="stretch")

    # --- Detalhes tecnicos ---
    with st.expander("Detalhes Tecnicos"):
        st.markdown(f"""
        **Modelo:** ResNet18 com fine-tuning
        **Dispositivo:** `{device}`
        **Tensor de entrada:** `{tuple(tensor.shape)}`
        **Probabilidade bruta:** `{result['probability']:.6f}`
        **Limiar utilizado:** `{threshold}`

        **Tecnicas de CV aplicadas:**
        - {'V' if use_clahe else 'X'} CLAHE
        - {'V' if use_gaussian else 'X'} Subtracao Gaussiana de Fundo
        - {'V' if use_circular else 'X'} Mascara Circular
        - {'V' if use_green else 'X'} Canal Verde
        """)

else:
    st.info("Faca upload de uma imagem de retina para comecar a analise.")

    st.subheader("Pipeline do Sistema")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**1. Entrada**\n\nUpload da imagem de retina (fundoscopia)")
    with col2:
        st.markdown("**2. Pre-processamento**\n\nCLAHE, Gaussiana, Mascara Circular, Resize")
    with col3:
        st.markdown("**3. Classificacao**\n\nResNet18 -> Normal ou Retinopatia")
    with col4:
        st.markdown("**4. Explicabilidade**\n\nGrad-CAM -> Heatmap das regioes ativadas")