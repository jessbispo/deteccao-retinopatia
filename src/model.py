"""
model.py
========
Definição do modelo de classificação de retinopatia diabética.
Usa ResNet18 pré-treinado com fine-tuning para classificação binária.
"""

import torch
import torch.nn as nn
from torchvision import models
import os


# ──────────────────────────────────────────────
# 1. ARQUITETURA DO MODELO
# ──────────────────────────────────────────────

class RetinopatiaModel(nn.Module):
    """
    Classificador binário de retinopatia diabética baseado em ResNet18.
    
    Estratégia de fine-tuning:
    - As camadas convolucionais iniciais (layer1, layer2) são CONGELADAS
      pois capturam features genéricas (bordas, texturas) que já são úteis.
    - As camadas finais (layer3, layer4, fc) são TREINÁVEIS
      pois precisam aprender features específicas de retina.
    - A camada fully-connected final é substituída por um classificador
      binário com dropout para regularização.
    
    Saída: logit único (usa BCEWithLogitsLoss no treino)
    """
    
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        super(RetinopatiaModel, self).__init__()
        
        # Carrega ResNet18 pré-treinada no ImageNet
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        
        # ── Congelar camadas iniciais ──────────────────────────
        if freeze_backbone:
            # Congela conv1, bn1, layer1 e layer2
            layers_to_freeze = [backbone.conv1, backbone.bn1,
                                 backbone.layer1, backbone.layer2]
            for layer in layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # ── Extrair o backbone (tudo exceto o classificador final) ──
        self.features = nn.Sequential(
            backbone.conv1,    # 7x7 conv, stride 2
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,  # stride 2 → 56x56
            backbone.layer1,   # 64 canais
            backbone.layer2,   # 128 canais
            backbone.layer3,   # 256 canais
            backbone.layer4,   # 512 canais
        )
        
        self.avgpool = backbone.avgpool  # Global Average Pooling → (512,)
        
        # ── Novo classificador binário ─────────────────────────
        # Substitui o FC original (1000 classes ImageNet) por classificação binária
        in_features = backbone.fc.in_features  # 512 para ResNet18
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),              # regularização
            nn.Linear(in_features, 256),    # camada intermediária
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1)               # saída binária (logit)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: tensor (batch, 3, 224, 224)
        Returns:
            logits: tensor (batch, 1) — sem sigmoid (aplicada no loss/predict)
        """
        x = self.features(x)          # (batch, 512, 7, 7)
        x = self.avgpool(x)            # (batch, 512, 1, 1)
        x = torch.flatten(x, 1)        # (batch, 512)
        x = self.classifier(x)         # (batch, 1)
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna probabilidade (0-1) de retinopatia."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> dict:
        """
        Retorna predição com label e confiança.
        Returns:
            dict com 'label', 'probability', 'confidence'
        """
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
        """Conta parâmetros treináveis."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Conta total de parâmetros."""
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────
# 2. FUNÇÕES UTILITÁRIAS
# ──────────────────────────────────────────────

def load_model(checkpoint_path: str = None,
               device: str = None) -> RetinopatiaModel:
    """
    Carrega o modelo. Se checkpoint_path fornecido, carrega pesos salvos.
    Caso contrário, retorna modelo com pesos ImageNet (para fine-tuning).
    
    Args:
        checkpoint_path: caminho para arquivo .pth salvo
        device: 'cuda', 'mps' ou 'cpu' (auto-detectado se None)
    Returns:
        modelo carregado e em modo eval
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"  # Apple Silicon
        else:
            device = "cpu"
    
    model = RetinopatiaModel(pretrained=(checkpoint_path is None))
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )
        # Suporta tanto state_dict direto quanto checkpoint completo
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"✓ Modelo carregado do checkpoint: {checkpoint_path}")
            if "epoch" in checkpoint:
                print(f"  → Epoch: {checkpoint['epoch']}, "
                      f"Val Acc: {checkpoint.get('val_acc', 'N/A'):.4f}")
        else:
            model.load_state_dict(checkpoint)
            print(f"✓ Modelo carregado: {checkpoint_path}")
    else:
        print("ℹ Usando pesos pré-treinados do ImageNet (sem fine-tuning de retina)")
    
    model = model.to(device)
    model.eval()
    return model, device


def get_loss_function():
    """
    BCEWithLogitsLoss com pos_weight para lidar com desbalanceamento de classes.
    
    Datasets de retinopatia são tipicamente desbalanceados (mais casos normais).
    pos_weight > 1 aumenta o peso dos casos positivos (retinopatia).
    Ajuste pos_weight = (n_negativos / n_positivos) do seu dataset.
    """
    # Exemplo: se dataset tem 70% normal e 30% retinopatia → pos_weight ≈ 2.3
    pos_weight = torch.tensor([2.3])
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def get_optimizer(model: RetinopatiaModel, lr: float = 1e-4):
    """
    AdamW com learning rates diferentes para backbone e classifier.
    O backbone (pré-treinado) recebe lr menor para não perder features úteis.
    """
    backbone_params = [p for n, p in model.named_parameters()
                       if "classifier" not in n and p.requires_grad]
    classifier_params = [p for n, p in model.named_parameters()
                         if "classifier" in n]
    
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.1},   # backbone: lr/10
        {"params": classifier_params, "lr": lr}         # classifier: lr normal
    ], weight_decay=1e-4)
