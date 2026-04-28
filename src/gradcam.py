"""
gradcam.py
Implementacao do Grad-CAM para visualizacao explicavel das predicoes.

Referencia: Selvaraju et al., 2017 - "Grad-CAM: Visual Explanations from
Deep Networks via Gradient-based Localization"

O Grad-CAM responde: quais regioes da imagem mais influenciaram a predicao?
Em aplicacoes medicas isso e essencial para dar confianca ao diagnostico
e verificar se o modelo esta olhando para as regioes certas da retina.

Funcionamento resumido:
  1. Forward pass: a imagem gera uma predicao
  2. Backward pass: gradientes da saida em relacao aos feature maps
     da ultima camada convolucional
  3. GAP dos gradientes -> peso de importancia por canal
  4. Combinacao ponderada dos feature maps -> mapa de ativacao (7x7)
  5. ReLU + normalizacao + upscale -> heatmap no tamanho da imagem
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple


class GradCAM:
    """
    Grad-CAM para modelos ResNet.

    Registra hooks no PyTorch para capturar ativacoes e gradientes
    da ultima camada convolucional (layer4 da ResNet18) sem precisar
    modificar a arquitetura do modelo.
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()

        # Por padrao usa a ultima conv do layer4, que captura features
        # de alto nivel e ainda tem resolucao espacial (7x7)
        if target_layer is None:
            target_layer = model.features[-1][-1].conv2

        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        """
        Registra um forward hook para salvar as ativacoes e um backward
        hook para salvar os gradientes durante o calculo do Grad-CAM.
        """
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        """Remove os hooks para liberar memoria apos o uso."""
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(self, input_tensor: torch.Tensor,
                 device: str = "cpu") -> Tuple[np.ndarray, float]:
        """
        Gera o mapa de ativacao Grad-CAM para uma imagem.

        Returns:
            cam  : heatmap normalizado (H, W) com valores em [0, 1]
            score: probabilidade de retinopatia
        """
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad_(True)

        self.model.zero_grad()
        output = self.model(input_tensor)
        score = torch.sigmoid(output).item()

        # Backward no logit para obter os gradientes na camada alvo
        output.backward()

        gradients = self.gradients    # (1, 512, 7, 7)
        activations = self.activations

        # GAP dos gradientes: peso de cada canal de feature map
        # alpha_k = media espacial dos gradientes do canal k
        weights = gradients.mean(dim=[2, 3], keepdim=True)  # (1, 512, 1, 1)

        # Combinacao linear ponderada: soma dos feature maps * pesos
        cam = (weights * activations).sum(dim=1, keepdim=True)

        # ReLU mantem apenas ativacoes positivas (que contribuem para a classe)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()

        # Normaliza para [0, 1] para aplicar o colormap
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
        Sobrepooe o heatmap Grad-CAM na imagem original.

        O colormap JET mapeia baixa ativacao para azul e alta para vermelho,
        facilitando a identificacao visual das regioes determinantes.

        Args:
            original_image: imagem RGB (H, W, 3)
            cam           : mapa Grad-CAM normalizado
            alpha         : peso do heatmap na mistura (0 = so original)
        """
        h, w = original_image.shape[:2]

        # Upscale com interpolacao bicubica para um resultado mais suave
        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)

        cam_uint8 = (cam_resized * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(cam_uint8, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        if original_image.dtype != np.uint8:
            original_image = (original_image * 255).astype(np.uint8)

        overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_rgb, alpha, 0)
        return overlay

    def generate_full_visualization(self, input_tensor: torch.Tensor,
                                     original_image_rgb: np.ndarray,
                                     device: str = "cpu") -> dict:
        """
        Gera todos os elementos visuais para exibicao na interface.

        Returns:
            dict com cam, cam_upscaled, heatmap, overlay, score, label,
            confidence e has_dr
        """
        cam, score = self.generate(input_tensor, device)

        h, w = original_image_rgb.shape[:2]
        cam_upscaled = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)

        cam_uint8 = (cam_upscaled * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        overlay = self.overlay_on_image(original_image_rgb, cam)

        has_dr = score >= 0.5
        label = "Retinopatia Detectada" if has_dr else "Normal"
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