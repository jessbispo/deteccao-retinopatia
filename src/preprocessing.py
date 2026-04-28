"""
preprocessing.py
Tecnicas de processamento de imagens fundoscopicas de retina.
"""

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


# ---------------------------------------------------------------------------
# Tecnicas individuais de processamento
# ---------------------------------------------------------------------------

def remove_black_border(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Remove a borda preta ao redor da retina.

    Imagens fundoscopicas sao capturadas com iluminacao circular, deixando
    cantos pretos sem informacao clinica. Remover essa borda evita que o
    modelo aprenda a partir desse fundo vazio.

    Tecnica: limiarizacao para separar pixels de conteudo dos pretos,
    seguida de bounding box do conteudo util.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Pixels acima do threshold sao considerados conteudo da retina
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(mask)
    if coords is None:
        return image

    x, y, w, h = cv2.boundingRect(coords)

    # Recorte quadrado centralizado para nao distorcer a proporcao
    size = max(w, h)
    cx, cy = x + w // 2, y + h // 2
    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)
    x2 = min(image.shape[1], x1 + size)
    y2 = min(image.shape[0], y1 + size)

    return image[y1:y2, x1:x2]


def apply_clahe(image: np.ndarray, clip_limit: float = 1.5,
                tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Melhora o contraste local da imagem com CLAHE.

    Imagens fundoscopicas tem iluminacao desigual, dificultando visualizar
    microaneurismas e exsudatos (lesoes da retinopatia). O CLAHE equaliza
    o histograma em blocos locais (tiles), realcando essas estruturas sem
    superexpor regioes ja claras.

    Aplicado no canal L do espaco LAB para atuar somente na luminancia
    e preservar as cores originais.
    """
    # Converte para LAB: L = luminancia, A/B = componentes de cor
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # clip_limit controla o quanto o contraste pode ser amplificado por tile
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def apply_gaussian_background_subtraction(image: np.ndarray, sigma: float = 30,
                                           strength: float = 0.3) -> np.ndarray:
    """
    Reduz variacoes de iluminacao de baixa frequencia (fundo nao uniforme).

    Cameras diferentes e angulos de captura geram iluminacao inconsistente
    entre imagens. Este filtro passa-alta estima o fundo via blur gaussiano
    e o subtrai parcialmente, normalizando o brilho e destacando detalhes
    finos como vasos e lesoes (metodo de Ben Graham, Kaggle APTOS 2019).

    O parametro strength controla a intensidade: valores altos podem gerar
    artefatos em imagens com iluminacao muito variada.
    """
    kernel_size = int(sigma * 2) | 1

    # O blur gaussiano estima o componente de baixa frequencia (o "fundo")
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    # Subtrai o fundo e centraliza o resultado em torno de 128
    corrected = cv2.addWeighted(image, 4, blurred, -4, 128)
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    # Blend parcial para evitar artefatos: mistura imagem original com a corrigida
    result = cv2.addWeighted(image, 1.0 - strength, corrected, strength, 0)
    return result


def extract_green_channel(image: np.ndarray) -> np.ndarray:
    """
    Isola o canal verde da imagem fundoscopica.

    A hemoglobina nos vasos sanguineos absorve fortemente a luz verde,
    tornando os vasos muito mais visiveis nesse canal do que no vermelho
    ou azul. Usado como alternativa ao CLAHE para enfatizar a estrutura
    vascular da retina.

    Retorna imagem de 3 canais identicos (todos = canal verde) para
    manter compatibilidade com redes treinadas em RGB.
    """
    green = image[:, :, 1]
    return cv2.merge([green, green, green])


def resize_with_padding(image: np.ndarray, target_size: int = 224) -> np.ndarray:
    """
    Redimensiona a imagem para tamanho fixo sem distorcer a proporcao.

    Redes convolucionais exigem entrada de tamanho fixo. Esticar a imagem
    diretamente distorceria as estruturas circulares da retina. O letterboxing
    preserva a proporcao adicionando padding preto onde necessario.
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Canvas preto do tamanho alvo; a imagem fica centralizada
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    return canvas


def apply_circular_mask(image: np.ndarray) -> np.ndarray:
    """
    Zera os pixels fora da regiao circular da retina.

    Apos o resize com padding, os cantos sao pretos mas ainda existem como
    pixels no tensor. A mascara circular garante que o modelo processe apenas
    a area de interesse diagnostico, evitando que aprenda com os cantos vazios.

    Tecnica: mascara binaria circular aplicada via AND bit a bit.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    radius = min(center) - 5

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    mask_3ch = cv2.merge([mask, mask, mask])
    return cv2.bitwise_and(image, mask_3ch)


def fill_background_with_mean(image: np.ndarray) -> np.ndarray:
    """
    Substitui o fundo preto pela cor media da retina.

    Alternativa a mascara preta: preenche os cantos com a cor media dos
    pixels da retina, reduzindo o contraste abrupto entre fundo e imagem
    que poderia ser aprendido como artefato pelo modelo.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    radius = min(center) - 5

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    mask_bool = mask.astype(bool)

    mean_color = image[mask_bool].mean(axis=0)

    result = image.copy()
    result[~mask_bool] = mean_color
    return result.astype(np.uint8)


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

def preprocess_retina_image(
    image_input,
    target_size: int = 224,
    use_clahe: bool = True,
    use_gaussian: bool = False,
    gaussian_strength: float = 0.0,
    use_green_channel: bool = False,
    use_circular_mask: bool = True,
    return_stages: bool = False
):
    """
    Pipeline de pre-processamento para imagens fundoscopicas de retina.

    Ordem das etapas:
      1. Leitura e conversao para BGR
      2. Remocao de borda preta
      3. Resize com padding (letterboxing)
      4. Mascara circular (opcional)
      5. CLAHE ou canal verde (opcional, mutuamente exclusivos)
      6. Subtracao gaussiana de fundo (opcional)
      7. Normalizacao ImageNet e conversao para tensor PyTorch

    Args:
        image_input      : PIL.Image, np.ndarray (BGR) ou caminho (str)
        target_size      : tamanho de saida quadrado (224 para ResNet)
        use_clahe        : aplica melhoria de contraste local
        use_gaussian     : aplica subtracao de fundo gaussiana
        gaussian_strength: intensidade da subtracao (0.0 a 1.0)
        use_green_channel: usa canal verde no lugar do CLAHE
        use_circular_mask: aplica mascara circular
        return_stages    : se True, retorna tambem um dict com cada etapa

    Returns:
        tensor PyTorch (1, 3, H, W) normalizado
        (e dict de etapas intermediarias se return_stages=True)
    """
    stages = {}

    # --- Leitura ---
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    elif isinstance(image_input, Image.Image):
        img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        raise ValueError("image_input deve ser str, PIL.Image ou np.ndarray")

    if return_stages:
        stages["0_original"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Etapa 1: remove borda preta ---
    img = remove_black_border(img)
    if return_stages:
        stages["1_sem_borda"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Etapa 2: resize sem distorcao ---
    img = resize_with_padding(img, target_size)
    if return_stages:
        stages["2_resize"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Etapa 3: isola regiao circular da retina ---
    if use_circular_mask:
        img = apply_circular_mask(img)
        if return_stages:
            stages["3_mascara_circular"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Etapa 4: melhoria de contraste ---
    if use_green_channel:
        img = extract_green_channel(img)
        if return_stages:
            stages["4_canal_verde"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        if use_clahe:
            img = apply_clahe(img)
            if return_stages:
                stages["4a_clahe"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if use_gaussian:
            img = apply_gaussian_background_subtraction(
                img, sigma=30, strength=gaussian_strength)
            if return_stages:
                stages["4b_gaussian"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Etapa 5: normalizacao para tensor PyTorch ---
    # Converte BGR para RGB (torchvision usa RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Media e desvio padrao do ImageNet, exigidos pelo modelo pre-treinado
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    tensor = transform(img_rgb).unsqueeze(0)

    if return_stages:
        stages["5_final_rgb"] = img_rgb
        return tensor, stages

    return tensor


# ---------------------------------------------------------------------------
# Transforms de augmentation para o loop de treino
# ---------------------------------------------------------------------------

def get_train_transforms(target_size: int = 224) -> transforms.Compose:
    """
    Augmentation aplicado durante o treino.

    Datasets medicos sao pequenos. Flips e rotacoes simulam diferentes
    orientacoes de captura (a retina pode ser fotografada em qualquer angulo).
    ColorJitter simula variacoes entre diferentes cameras e condicoes de luz.
    """
    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms(target_size: int = 224) -> transforms.Compose:
    """
    Transforms de validacao e teste (sem augmentation).
    """
    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])