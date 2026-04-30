"""
model.py
Definicao, carregamento e utilitarios do modelo ResNet18 para
classificacao binaria de retinopatia diabetica.
"""

import torch
import torch.nn as nn
from torchvision import models
import os


# ---------------------------------------------------------------------------
# Arquitetura
# ---------------------------------------------------------------------------

class RetinopatiaModel(nn.Module):
    """
    Classificador binario de retinopatia baseado em ResNet18 pre-treinada.

    Estrategia de fine-tuning em duas fases:
      - Fase 1: backbone congelado, apenas o classificador e treinado.
        Isso estabiliza o aprendizado inicial sem destruir os pesos do ImageNet.
      - Fase 2 (metade do treino): backbone descongelado com lr reduzido,
        permitindo ajuste fino das features para o dominio de retina.

    A camada FC original (1000 classes ImageNet) e substituida por um
    classificador binario com dropout para regularizacao.
    """

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        super(RetinopatiaModel, self).__init__()

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Congela as camadas iniciais: capturam bordas e texturas genericas
        # que ja sao uteis e nao precisam ser re-aprendidas
        if freeze_backbone:
            for layer in [backbone.conv1, backbone.bn1,
                          backbone.layer1, backbone.layer2]:
                for param in layer.parameters():
                    param.requires_grad = False

        # Extrai o backbone completo exceto o classificador original
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,   # 64 canais
            backbone.layer2,   # 128 canais
            backbone.layer3,   # 256 canais
            backbone.layer4,   # 512 canais
        )

        self.avgpool = backbone.avgpool

        # Classificador binario com dropout para reduzir overfitting
        in_features = backbone.fc.in_features  # 512 na ResNet18
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1)   # saida unica: logit de retinopatia
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)       # (batch, 512, 7, 7)
        x = self.avgpool(x)        # (batch, 512, 1, 1)
        x = torch.flatten(x, 1)   # (batch, 512)
        x = self.classifier(x)    # (batch, 1)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna probabilidade de retinopatia entre 0 e 1."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> dict:
        """Retorna label, probabilidade e confianca para uma imagem."""
        prob = self.predict_proba(x).item()
        label = "Retinopatia Detectada" if prob >= threshold else "Normal"
        confidence = prob if prob >= threshold else 1 - prob
        return {
            "label": label,
            "probability": prob,
            "confidence": confidence,
            "has_dr": prob >= threshold
        }

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str = None,
               device: str = None):
    """
    Carrega o modelo. Se um checkpoint for fornecido, restaura os pesos
    salvos. Caso contrario, usa os pesos pre-treinados do ImageNet.

    Args:
        checkpoint_path: caminho para o arquivo .pth salvo durante o treino
        device         : 'cuda', 'mps' ou 'cpu' (auto-detectado se None)
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = RetinopatiaModel(pretrained=(checkpoint_path is None))

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device,
                                weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Modelo carregado: {checkpoint_path}")
            if "epoch" in checkpoint:
                print(f"  Epoch: {checkpoint['epoch']} | "
                      f"Val Acc: {checkpoint.get('val_acc', 0):.4f}")
        else:
            model.load_state_dict(checkpoint)
            print(f"Modelo carregado: {checkpoint_path}")
    else:
        print("Usando pesos ImageNet (sem fine-tuning de retina)")

    model = model.to(device)
    model.eval()
    return model, device


# ---------------------------------------------------------------------------
# Loss e otimizador
# ---------------------------------------------------------------------------

def get_loss_function():
    """
    BCEWithLogitsLoss com pos_weight para lidar com desbalanceamento.

    Datasets de retinopatia tem mais casos normais do que positivos.
    pos_weight = (n_normal / n_retinopatia) aumenta a penalidade por
    errar casos de retinopatia, forcando o modelo a aprende-los.
    O valor correto e calculado automaticamente no train.py.
    """
    pos_weight = torch.tensor([2.3])
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def get_optimizer(model: RetinopatiaModel, lr: float = 1e-4):
    """
    AdamW com learning rates diferentes para backbone e classificador.

    O backbone ja tem bons pesos do ImageNet, entao recebe lr/10 para
    nao perder essas features. O classificador, treinado do zero, recebe
    o lr completo para aprender mais rapido.
    """
    backbone_params = [p for n, p in model.named_parameters()
                       if "classifier" not in n and p.requires_grad]
    classifier_params = [p for n, p in model.named_parameters()
                         if "classifier" in n]

    return torch.optim.AdamW([
        {"params": backbone_params,    "lr": lr * 0.1},
        {"params": classifier_params,  "lr": lr}
    ], weight_decay=1e-4)