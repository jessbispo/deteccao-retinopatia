# Classificacao de Retinopatia Diabetica com Visualizacao Explicavel

**Universidade Presbiteriana Mackenzie - Computacao Visual**

- Bruna Aguiar Muchiuti
- Gabriel Ken Kazama Geronazzo
- Jessica dos Santos Santana Bispo
- Lucas Pires de Camargo Sarai

---

## Sobre o Projeto

Aplicacao de visao computacional para classificacao binaria de retinopatia
diabetica em imagens de retina (fundoscopia), com visualizacao explicavel
via Grad-CAM.

O diagnostico de retinopatia depende de especialistas analisando imagens
fundoscopicas, processo demorado e dependente de disponibilidade clinica.
Este projeto automatiza parte desse processo usando uma CNN pre-treinada
com fine-tuning no dominio de retina.

---

## Tecnicas de Computacao Visual

### Pre-processamento (`preprocessing.py`)

| Tecnica | Funcao | Por que usamos |
|---|---|---|
| Thresholding + BBox | `remove_black_border` | Remove borda preta circular tipica de fundoscopia |
| Letterboxing | `resize_with_padding` | Redimensiona sem distorcer estruturas circulares |
| Mascara Circular | `apply_circular_mask` | Foca o modelo na regiao de interesse diagnostico |
| CLAHE | `apply_clahe` | Realca microaneurismas e exsudatos localmente |
| Subtracao Gaussiana | `apply_gaussian_background_subtraction` | Remove iluminacao nao uniforme entre cameras |
| Canal Verde | `extract_green_channel` | Maximo contraste para vasos sanguineos |

### Modelo (`model.py`)

- ResNet18 pre-treinada no ImageNet com fine-tuning em duas fases
- Fase 1: backbone congelado, apenas o classificador e treinado
- Fase 2 (metade do treino): backbone descongelado com lr reduzido
- Classificador binario substituindo o FC original do ImageNet

### Explicabilidade (`gradcam.py`)

- Grad-CAM (Selvaraju et al., 2017)
- Calcula gradientes da saida em relacao aos feature maps da ultima camada convolucional
- Gera heatmap mostrando as regioes determinantes para o diagnostico

---

## Estrutura do Projeto

```
retinopatia-deteccao/
├── data/
│   ├── processed/
│      ├── train/
│      │   ├── normal/
│      │   └── retinopatia/
│      └── val/
│          ├── normal/
│          └── retinopatia/
│   
├── src/
│   ├── app.py              # Interface Streamlit
│   ├── model.py            # Arquitetura ResNet18 e utilitarios
│   ├── preprocessing.py    # Pipeline de pre-processamento
│   ├── gradcam.py          # Visualizacao Grad-CAM
│   └── train.py            # Script de fine-tuning
├── modelo_retina.pth       # Pesos salvos (gerado apos o treino)
├── requirements.txt
└── README.md
```

---

## Como Executar

### 1. Instalar dependencias

**Criar ambiente virtual para dependências não ficarem fixas na máquina:**
```bash
python3 -m venv .venv;
source .venv/bin/activate
```
**Baixar as dependências dentro do ambiente virtual:**
```bash
python3 -m pip install -r requirements.txt
```


### 2. Preparar o dataset

O projeto usa o dataset **APTOS 2019** disponivel no Kaggle:
https://www.kaggle.com/competitions/aptos2019-blindness-detection

Apos baixar, organize as imagens conforme abaixo.
Imagens com `diagnosis == 0` vao para `normal/`,
imagens com `diagnosis >= 1` vao para `retinopatia/`.

```
data/processed/
├── train/
│   ├── normal/
│   └── retinopatia/
└── val/
    ├── normal/
    └── retinopatia/
```

### 3. Treinar o modelo *(Opcional)*

```bash
python src/train.py --data_dir ./data/processed --epochs 20 --batch_size 16
```

Parametros disponiveis:

| Parametro | Padrao | Descricao |
|---|---|---|
| `--data_dir` | `./data/processed` | Diretorio do dataset |
| `--epochs` | `20` | Numero de epocas |
| `--batch_size` | `16` | Tamanho do batch |
| `--lr` | `1e-4` | Learning rate inicial |
| `--save_path` | `modelo_retina.pth` | Caminho para salvar o modelo |

### 4. Executar a interface

```bash
streamlit run src/app.py
```

Se `modelo_retina.pth` nao existir, a aplicacao carrega os pesos
pre-treinados do ImageNet. Nesse caso os resultados nao sao confiaveis
clinicamente pois o modelo nao foi ajustado para retina.

---

## Fluxo do Sistema

```
<<<<<<< HEAD
Imagem de Retina (fundoscopia)
         |
         v
+------------------------------------------+
|            Pre-processamento             |
|  1. Remove borda preta (Thresholding)    |
|  2. Resize com padding (Letterboxing)    |
|  3. Mascara circular (ROI)               |
|  4. CLAHE (contraste adaptativo)         |
|  5. Subtracao Gaussiana (fundo)          |
|  6. Normalizacao ImageNet                |
+------------------------------------------+
         |
         v
+------------------------------------------+
|          ResNet18 (Fine-tuned)           |
|  conv1, layer1, layer2  (congelados)    |
|  layer3, layer4, FC     (treinaveis)    |
|  saida: logit -> sigmoid -> prob [0,1]  |
+------------------------------------------+
         |
         v
+------------------------------------------+
|               Grad-CAM                   |
|  1. Forward hook -> salva ativacoes      |
|  2. Backward hook -> salva gradientes    |
|  3. GAP dos gradientes -> pesos          |
|  4. Combinacao linear -> CAM (7x7)       |
|  5. ReLU + normalizacao + upscale        |
+------------------------------------------+
         |
         v
  Classificacao + Heatmap sobreposto
=======
Imagem de Retina
      │
      ▼
┌─────────────────────────────────────────┐
│           Pré-processamento             │
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
│         ResNet18 (Fine-tuned)           │
│  conv1 → layer1 → layer2 (congelados)   │
│  layer3 → layer4 → FC (treináveis)      │
│  Saída: logit → sigmoid → probabilidade │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│              Grad-CAM                   │
│  1. Captura ativações (forward hook)    │
│  2. Captura gradientes (backward hook)  │
│  3. GAP dos gradientes → pesos          │
│  4. Combinação linear → CAM (7×7)       │
│  5. ReLU + Normalização + Upscale       │
└─────────────────────────────────────────┘
      │
      ▼
  Resultado + Heatmap sobreposto
>>>>>>> dc053ded5fd174eea47264f30d7e1961fa53f1cf
```

---

## Tecnologias

| Biblioteca | Uso |
|---|---|
| PyTorch | Modelo, treino e Grad-CAM |
| torchvision | ResNet18 pre-treinada e transforms |
| OpenCV | CLAHE, filtros gaussianos, mascaras |
| Streamlit | Interface web |
| NumPy / Matplotlib | Visualizacoes e manipulacao de arrays |

---

## Aviso

Este projeto e de natureza academica e nao deve ser usado para
diagnostico medico real. Sempre consulte um especialista qualificado.
