"""
gradcam.py
==========
Implementação do Grad-CAM (Gradient-weighted Class Activation Mapping).

Referência: Selvaraju et al., 2017 — "Grad-CAM: Visual Explanations from
Deep Networks via Gradient-based Localization"

O Grad-CAM responde: "Quais regiões da imagem mais influenciaram a predição?"
Isso é crucial em aplicações médicas para dar confiança ao diagnóstico.

Como funciona:
1. Forward pass: a imagem passa pela rede e gera uma predição
2. Backprop parcial: calcula gradientes da saída em relação aos
   feature maps da última camada convolucional
3. Global Average Pooling dos gradientes → pesos de importância por canal
4. Combinação linear dos feature maps com esses pesos → mapa de ativação
5. ReLU + normalização + upscale → heatmap do tamanho da imagem original
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple


class GradCAM:
    """
    Grad-CAM para modelos ResNet.
    
    Usa hooks do PyTorch para capturar ativações e gradientes
    da última camada convolucional (layer4 da ResNet18).
    """
    
    def __init__(self, model, target_layer=None):
        """
        Args:
            model: RetinopatiaModel
            target_layer: camada alvo (padrão: última camada de layer4)
        """
        self.model = model
        self.model.eval()
        
        # Usa a última camada convolucional do backbone por padrão
        if target_layer is None:
            # Para ResNet18: features[-1] é o layer4, [-1] é o último BasicBlock
            target_layer = model.features[-1][-1].conv2
        
        self.target_layer = target_layer
        
        # Armazena ativações e gradientes via hooks
        self.activations = None
        self.gradients = None
        
        # Registra hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """
        Registra forward hook (captura ativações) e
        backward hook (captura gradientes).
        """
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
    
    def remove_hooks(self):
        """Remove os hooks para liberar memória."""
        self.forward_handle.remove()
        self.backward_handle.remove()
    
    def generate(self, input_tensor: torch.Tensor,
                 device: str = "cpu") -> Tuple[np.ndarray, float]:
        """
        Gera o mapa de ativação Grad-CAM.
        
        Args:
            input_tensor: tensor (1, 3, H, W) — imagem pré-processada
            device: dispositivo de processamento
        
        Returns:
            cam: heatmap normalizado (H, W) com valores [0, 1]
            score: probabilidade de retinopatia (0-1)
        """
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad_(True)
        
        # ── 1. Forward pass ───────────────────────────────────
        self.model.zero_grad()
        output = self.model(input_tensor)          # (1, 1) logit
        score = torch.sigmoid(output).item()       # probabilidade
        
        # ── 2. Backward pass ──────────────────────────────────
        # Para classificação binária, fazemos backward no logit diretamente
        output.backward()
        
        # ── 3. Calcula pesos via Global Average Pooling dos gradientes ─
        # gradients shape: (1, C, H', W') onde C=512 para ResNet18 layer4
        gradients = self.gradients  # (1, 512, 7, 7)
        activations = self.activations  # (1, 512, 7, 7)
        
        # GAP dos gradientes → peso de cada canal
        # α_k = (1/Z) * Σ_ij (∂y / ∂A^k_ij)
        weights = gradients.mean(dim=[2, 3], keepdim=True)  # (1, 512, 1, 1)
        
        # ── 4. Combinação linear ponderada dos feature maps ───
        # L_Grad-CAM = ReLU(Σ_k α_k * A^k)
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, 7, 7)
        
        # ReLU: mantém apenas ativações positivas (que contribuem para a classe)
        cam = F.relu(cam)
        
        # ── 5. Normalização e upscale ─────────────────────────
        cam = cam.squeeze().cpu().numpy()  # (7, 7)
        
        # Normaliza para [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        
        return cam, score
    
    def overlay_on_image(self, original_image: np.ndarray,
                          cam: np.ndarray,
                          alpha: float = 0.4,
                          colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Sobrepõe o heatmap Grad-CAM na imagem original.
        
        Processo:
        1. Upscale do CAM (7x7 → 224x224) com interpolação bicúbica
        2. Aplica colormap (JET: azul→verde→vermelho = baixa→alta ativação)
        3. Blend com a imagem original usando transparência
        
        Args:
            original_image: imagem RGB (H, W, 3) — imagem pré-processada
            cam: mapa Grad-CAM normalizado (H', W')
            alpha: peso do heatmap na mistura (0=só original, 1=só heatmap)
            colormap: colormap OpenCV (JET é o mais comum para heatmaps)
        
        Returns:
            imagem RGB com heatmap sobreposto
        """
        h, w = original_image.shape[:2]
        
        # Upscale com interpolação bicúbica (mais suave que nearest)
        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Converte para uint8 [0, 255] e aplica colormap
        cam_uint8 = (cam_resized * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(cam_uint8, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        
        # Garante que a imagem original está em uint8
        if original_image.dtype != np.uint8:
            original_image = (original_image * 255).astype(np.uint8)
        
        # Blend: resultado = (1-alpha)*original + alpha*heatmap
        overlay = cv2.addWeighted(original_image, 1 - alpha,
                                   heatmap_rgb, alpha, 0)
        
        return overlay
    
    def generate_full_visualization(
        self,
        input_tensor: torch.Tensor,
        original_image_rgb: np.ndarray,
        device: str = "cpu"
    ) -> dict:
        """
        Gera a visualização completa para exibição na interface.
        
        Returns dict com:
        - 'cam': mapa de ativação raw (7x7, normalizado)
        - 'cam_upscaled': CAM redimensionado (224x224)
        - 'heatmap': heatmap colorido (224x224 RGB)
        - 'overlay': imagem original + heatmap (224x224 RGB)
        - 'score': probabilidade de retinopatia
        - 'label': texto da predição
        - 'confidence': confiança da predição
        """
        cam, score = self.generate(input_tensor, device)
        
        h, w = original_image_rgb.shape[:2]
        cam_upscaled = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Heatmap colorido isolado
        cam_uint8 = (cam_upscaled * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = self.overlay_on_image(original_image_rgb, cam)
        
        # Label
        has_dr = score >= 0.5
        label = "⚠️ Retinopatia Detectada" if has_dr else "✅ Normal"
        confidence = score if has_dr else 1 - score
        
        return {
            "cam": cam,
            "cam_upscaled": cam_upscaled,
            "heatmap": heatmap_rgb,
            "overlay": overlay,
            "score": score,
            "label": label,
            "confidence": confidence,
            "has_dr": has_dr
        }
