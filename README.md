# Classificação de Retinopatia Diabética com Visualização Explicável

**Universidade Presbiteriana Mackenzie — Computação Visual**

- Bruna Aguiar Muchiuti 
- Gabriel Ken Kazama Geronazzo 
- Jessica dos Santos Santana Bispo 
- Lucas Pires de Camargo Sarai

---

## Sobre

Aplicação de visão computacional para classificação binária de retinopatia diabética em imagens de retina (fundoscopia), com visualização explicável via Grad-CAM.

---

## Técnicas de Computação Visual

### Pré-processamento (`preprocessing.py`)

| Técnica | Função | Por que usamos |
|---|---|---|
| **Thresholding + BBox** | `remove_black_border()` | Remove borda preta circular típica de fundoscopia |
| **Letterboxing** | `resize_with_padding()` | Redimensiona sem distorção geométrica |
| **Máscara Circular** | `circular_crop()` | Foca apenas na ROI diagnóstica |
| **CLAHE** | `clahe_enhancement()` | Realça microaneurismas e exsudatos localmente |
| **Gaussian BG Subtraction** | `gaussian_preprocessing()` | Remove iluminação não uniforme (Ben Graham's method) |
| **Canal Verde** | `green_channel_extraction()` | Máximo contraste para vasos sanguíneos |

### Modelo (`model.py`)
- **ResNet18** pré-treinada no ImageNet com fine-tuning
- Camadas iniciais congeladas (features genéricas)
- Camadas finais treináveis (features de retina)
- Fine-tuning progressivo: descongelamento na metade do treino

### Explicabilidade (`gradcam.py`)
- **Grad-CAM**: calcula gradientes da saída em relação aos feature maps da última camada convolucional
- Gera heatmap que mostra as regiões determinantes para o diagnóstico

---

## Estrutura do Projeto

```

├───data
│   ├───processed
│   │   ├───train
│   │   │   ├───normal
│   │   │   └───retinopatia
│   │   └───val
│   │       ├───normal
│   │       └───retinopatia
│   └───raw
│       ├───test_images
│       │   └───test_images
│       ├───train_images
│       │   └───train_images
│       └───val_images
│           └───val_images
├───src
│   ├───app.py # Interface Streamlit
│   ├───model.py # Modelo ResNet18
│   ├───preprocessing.py # Pipeline de pré-processamento (CV)
│   ├───gradcam.py  # Visualização Grad-CAM
│   ├───train.py 
├── requirements.txt
└── README.md
```

---

## Como Executar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Preparar o dataset

Organize o dataset na estrutura abaixo:

```

├───data
│   ├───processed
│   │   ├───train
│   │   │   ├───normal # imagens sem retinopatia (label 0)
│   │   │   └───retinopatia # imagens com retinopatia (label 1)
│   │   └───val
│   │       ├───normal
│   │       └───retinopatia

```

**Datasets Utilizado:**
- [APTOS 2019](https://www.kaggle.com/competitions/aptos2019-blindness-detection) — Kaggle

> Para o APTOS: `diagnosis == 0` → `normal/`, `diagnosis >= 1` → `retinopatia/`

### 3. Treinar o modelo

```bash
python src/train.py --data_dir ./data/processed --epochs 20 --batch_size 32
```

Parâmetros disponíveis:
```
--data_dir        Diretório do dataset (padrão: ./dataset)
--epochs          Número de épocas (padrão: 20)
--batch_size      Tamanho do batch (padrão: 32)
--lr              Learning rate (padrão: 1e-4)
--save_path       Onde salvar o modelo (padrão: modelo_retina.pth)
```

### 4. Executar a interface

```bash
streamlit run src/app.py
```

> Se não houver `modelo_retina.pth`, a aplicação usa os pesos pré-treinados do ImageNet (sem fine-tuning de retina — resultados não confiáveis clinicamente).

---

## Tecnologias

- **Python 3.10+**
- **PyTorch** — modelo e treino
- **torchvision** — ResNet18 pré-treinada
- **OpenCV** — processamento de imagens (CLAHE, Gaussian, máscaras)
- **Streamlit** — interface web
- **NumPy / Matplotlib** — visualizações

---

## Fluxo do Sistema

```
Imagem de Retina
      │
      ▼
┌─────────────────────────────────────────┐
│           Pré-processamento              │
│  1. Remove borda preta (Thresholding)   │
│  2. Resize com padding (Letterboxing)   │
│  3. Máscara circular (ROI)              │
│  4. CLAHE (contraste adaptativo)        │
│  5. Gaussian BG Subtraction             │
│  6. Normalização ImageNet               │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│         ResNet18 (Fine-tuned)            │
│  conv1 → layer1 → layer2 (congelados)  │
│  layer3 → layer4 → FC (treináveis)     │
│  Saída: logit → sigmoid → probabilidade │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│              Grad-CAM                    │
│  1. Captura ativações (forward hook)    │
│  2. Captura gradientes (backward hook)  │
│  3. GAP dos gradientes → pesos          │
│  4. Combinação linear → CAM (7×7)       │
│  5. ReLU + Normalização + Upscale       │
└─────────────────────────────────────────┘
      │
      ▼
  Resultado + Heatmap sobreposto
```

---

## Aviso

Esta aplicação é um projeto acadêmico e **não deve ser utilizada para diagnóstico médico real**. Sempre consulte um especialista qualificado.
