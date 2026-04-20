"""
preprocessing.py
================
Módulo de pré-processamento de imagens de retina.
Foco em Computação Visual: cada etapa é uma técnica clássica de processamento
de imagens aplicada ao domínio médico.
"""

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


# ──────────────────────────────────────────────
# 1. TÉCNICAS DE PROCESSAMENTO DE IMAGEM (CV)
# ──────────────────────────────────────────────

def remove_black_border(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Remove a borda preta circular típica de imagens de retina (fundoscopia).
    
    Técnica: limiarização (thresholding) + bounding box do conteúdo útil.
    As imagens de retina geralmente têm um fundo preto ao redor do disco óptico,
    que não contribui para a classificação e pode atrapalhar a normalização.
    
    Args:
        image: imagem BGR (numpy array)
        threshold: valor mínimo de intensidade para considerar "conteúdo"
    Returns:
        imagem recortada sem bordas pretas excessivas
    """
    # Converte para escala de cinza 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Limiarização: pixels com valor > threshold são considerados "conteudo"
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Encontra o bounding box do conteúdo útil
    coords = cv2.findNonZero(mask)
    if coords is None:
        return image  # imagem toda preta, retorna original
    
    x, y, w, h = cv2.boundingRect(coords)
    
    # Garante recorte quadrado centralizado
    size = max(w, h)
    cx, cy = x + w // 2, y + h // 2
    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)
    x2 = min(image.shape[1], x1 + size)
    y2 = min(image.shape[0], y1 + size)
    
    return image[y1:y2, x1:x2]


def clahe_enhancement(image: np.ndarray, clip_limit: float = 2.0,
                       tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Técnica fundamental em visão médica: equaliza o histograma de forma local,
    melhorando o contraste em regiões específicas sem superexpor outras.
    Especialmente útil para realçar microaneurismas e exsudatos na retina,
    que são sinais diagnósticos da retinopatia diabética.
    
    Diferença do HE simples: o CLAHE opera em tiles (blocos locais) e
    limita o contraste para evitar amplificação de ruído.
    
    Args:
        image: imagem BGR
        clip_limit: limite de contraste (2.0 é um bom padrão para retina)
        tile_grid_size: tamanho dos blocos locais
    Returns:
        imagem com contraste melhorado no canal L (espaço LAB)
    """
    # Converte BGR → LAB para aplicar CLAHE apenas na luminancia (L)
    # Isso preserva as cores originais e evita distorções cromáticas
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Aplica CLAHE no canal de luminancia
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)
    
    # Reconstroi a imagem LAB com o L melhorado
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    
    # Converte de volta para BGR
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def green_channel_extraction(image: np.ndarray) -> np.ndarray:
    """
    Extrai e realça o canal verde da imagem de retina.
    
    Técnica específica de domínio: o canal verde (G) das imagens de fundoscopia
    apresenta o maior contraste para estruturas vasculares da retina.
    Isso ocorre porque a hemoglobina absorve fortemente no verde, tornando
    os vasos sanguíneos muito mais visíveis nesse canal.
    
    Retorna imagem BGR onde os 3 canais são o canal verde original,
    para manter compatibilidade com redes pré-treinadas em RGB.
    
    Args:
        image: imagem BGR
    Returns:
        imagem BGR com os 3 canais substituídos pelo canal verde
    """
    green = image[:, :, 1]  # Canal G no espaço BGR
    # Empilha 3x para manter formato (H, W, 3)
    return cv2.merge([green, green, green])


def gaussian_preprocessing(image: np.ndarray, sigma: float = 10) -> np.ndarray:
    """
    Técnica de subtração de fundo com filtro Gaussiano (Ben Graham's method).
    
    Método popular em competições de detecção de retinopatia (ex: Kaggle APTOS).
    Remove variações de iluminação de baixa frequência (fundo não uniforme)
    preservando detalhes de alta frequência (vasos, lesões).
    
    Operação: resultado = original * α + blur * β + γ
    onde blur é a versão suavizada (fundo estimado).
    
    Args:
        image: imagem BGR
        sigma: desvio padrão do filtro Gaussiano (controla o "tamanho" do fundo)
    Returns:
        imagem com fundo normalizado
    """
    # Tamanho do kernel (deve ser ímpar)
    kernel_size = int(sigma * 4) | 1  # força valor ímpar com OR 1
    
    # Blur Gaussiano estima o "fundo" (iluminação não uniforme)
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    # Subtrai o fundo e adiciona offset de 128 para centralizar
    # addWeighted: dst = src1*alpha + src2*beta + gamma
    result = cv2.addWeighted(image, 4, blurred, -4, 128)
    
    return result


def resize_with_padding(image: np.ndarray, target_size: int = 224) -> np.ndarray:
    """
    Redimensiona a imagem mantendo a proporção, com padding preto se necessário.
    
    Técnica: letterboxing — comum em visão computacional para evitar
    distorção geométrica ao forçar uma proporção fixa. 
    
    Args:
        image: imagem BGR
        target_size: tamanho alvo (quadrado)
    Returns:
        imagem redimensionada para target_size x target_size
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Cria canvas preto e centraliza a imagem
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    
    return canvas


def circular_crop(image: np.ndarray) -> np.ndarray:
    """
    Aplica uma máscara circular para remover cantos da imagem.
    
    As imagens de retina têm formato circular. Os cantos (fora do círculo)
    são ruído que pode confundir o modelo. Aplicar uma máscara circular
    foca o modelo apenas na região de interesse diagnóstico.
    
    Técnica: criação de máscara binária circular + multiplicação elemento a elemento.
    
    Args:
        image: imagem quadrada (H == W)
    Returns:
        imagem com cantos zerados (preto)
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    radius = min(center) - 5  # pequena margem
    
    # Cria máscara circular branca sobre fundo preto
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)  # -1 = preenchido
    
    # Aplica a máscara nos 3 canais
    mask_3ch = cv2.merge([mask, mask, mask])
    return cv2.bitwise_and(image, mask_3ch)


# ──────────────────────────────────────────────
# 2. PIPELINE COMPLETO DE PRÉ-PROCESSAMENTO
# ──────────────────────────────────────────────

def preprocess_retina_image(
    image_input,
    target_size: int = 224,
    use_clahe: bool = True,
    use_gaussian: bool = True,
    use_green_channel: bool = False,
    use_circular_mask: bool = True,
    return_stages: bool = False
):
    """
    Pipeline completo de pré-processamento de imagem de retina.
    
    Ordem das etapas:
    1. Leitura e conversão para BGR (OpenCV)
    2. Remoção de borda preta
    3. Redimensionamento com padding
    4. Máscara circular (opcional)
    5. CLAHE - melhoria de contraste local (opcional)
    6. Gaussian background subtraction (opcional)
    7. Extração canal verde (opcional, alternativo ao CLAHE+Gaussian)
    8. Normalização e conversão para tensor PyTorch
    
    Args:
        image_input: PIL.Image, numpy array (BGR) ou path (str)
        target_size: tamanho alvo (padrão 224 para ResNet)
        use_clahe: aplica CLAHE
        use_gaussian: aplica subtração Gaussiana de fundo
        use_green_channel: usa apenas canal verde
        use_circular_mask: aplica máscara circular
        return_stages: se True, retorna dict com cada etapa (para visualização)
    
    Returns:
        tensor PyTorch shape (1, 3, target_size, target_size) normalizado
        (e dict de etapas se return_stages=True)
    """
    stages = {}
    
    # ── Etapa 0: Leitura ──────────────────────────────────────
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    elif isinstance(image_input, Image.Image):
        img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
        if img.shape[2] == 3 and img[0, 0, 0] != img[0, 0, 2]:
            pass  # assume BGR
    else:
        raise ValueError("image_input deve ser str, PIL.Image ou np.ndarray")
    
    if return_stages:
        stages["0_original"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # ── Etapa 1: Remove borda preta ───────────────────────────
    img = remove_black_border(img)
    if return_stages:
        stages["1_sem_borda"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # ── Etapa 2: Resize com padding ───────────────────────────
    img = resize_with_padding(img, target_size)
    if return_stages:
        stages["2_resize"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # ── Etapa 3: Máscara circular ─────────────────────────────
    if use_circular_mask:
        img = circular_crop(img)
        if return_stages:
            stages["3_mascara_circular"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # ── Etapa 4: Melhorias de contraste ───────────────────────
    if use_green_channel:
        img = green_channel_extraction(img)
        if return_stages:
            stages["4_canal_verde"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        if use_clahe:
            img = clahe_enhancement(img)
            if return_stages:
                stages["4a_clahe"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if use_gaussian:
            img = gaussian_preprocessing(img)
            if return_stages:
                stages["4b_gaussian"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # ── Etapa 5: Normalização para tensor PyTorch ─────────────
    # Converte BGR → RGB (PyTorch/torchvision usa RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Transformação padrão ImageNet (exigida pelos modelos pré-treinados)
    transform = transforms.Compose([
        transforms.ToTensor(),                          # [0,255] → [0,1]  shape: (3, H, W)
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],                # médias do ImageNet
            std=[0.229, 0.224, 0.225]                  # desvios do ImageNet
        )
    ])
    
    tensor = transform(img_rgb).unsqueeze(0)  # adiciona batch dim: (1, 3, H, W)
    
    if return_stages:
        stages["5_final_rgb"] = img_rgb
        return tensor, stages
    
    return tensor


# ──────────────────────────────────────────────
# 3. DATA AUGMENTATION (para treino)
# ──────────────────────────────────────────────

def get_train_transforms(target_size: int = 224) -> transforms.Compose:
    """
    Transformações de data augmentation para o conjunto de treino.
    
    Técnicas aplicadas:
    - Flip horizontal/vertical: a retina pode estar em qualquer orientação
    - Rotação aleatória: invariância à rotação do olho
    - Ajuste de brilho/contraste: simula diferentes condições de captura
    - ColorJitter: variações de câmera e iluminação
    
    O augmentation é fundamental pois datasets médicos são pequenos.
    """
    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms(target_size: int = 224) -> transforms.Compose:
    """
    Transformações para validação/teste (sem augmentation).
    """
    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
