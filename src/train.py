"""
Treino de modelo (ResNet18) para classificação bináriade retinopatia diabética.

Entrada esperada:
data/
    processed/
    ├── train/
    │   ├── normal/
    │   └── retinopatia/
    └── val/
        ├── normal/
        └── retinopatia/
"""

import os
import argparse
import time
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2
import numpy as np
from PIL import Image

from model import RetinopatiaModel, get_loss_function, get_optimizer
from preprocessing import get_train_transforms, get_val_transforms, preprocess_retina_image


# ──────────────────────────────────────────────
# 1. DATASET CUSTOMIZADO
# ──────────────────────────────────────────────

class RetinaDataset(Dataset):
    """
        Dataset de imagens de retina com pré-processamento
        (CLAHE, Gaussian, máscara circular).
    """
     
    def __init__(self, root_dir: str, split: str = "train",
                 target_size: int = 224, use_cv_preprocessing: bool = True):
    
        self.root_dir = os.path.join(root_dir, split)
        self.target_size = target_size
        self.use_cv_preprocessing = use_cv_preprocessing
        self.split = split
        
        self.samples = []
        self.class_to_idx = {"normal": 0, "retinopatia": 1}
        
        for class_name, label in self.class_to_idx.items():
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Diretorio não encontrado: {class_dir}")
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp')):
                    self.samples.append((
                        os.path.join(class_dir, fname),
                        label
                    ))
        
        # Transforms de augmentation (só para treino)
        if split == "train":
            self.augment = v2.Compose([
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation(degrees=30),
                v2.ColorJitter(brightness=0.2, contrast=0.2),
            ])
        else:
            self.augment = None
        
        print(f"Dataset '{split}': {len(self.samples)} imagens "
              f"({sum(1 for _, l in self.samples if l == 0)} normal, "
              f"{sum(1 for _, l in self.samples if l == 1)} retinopatia)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Carrega como PIL para compatibilidade
        img_pil = Image.open(img_path).convert("RGB")
        
        if self.use_cv_preprocessing:
            # Aplica pipeline de CV (CLAHE, Gaussian, etc.)
            tensor = preprocess_retina_image(
                img_pil,
                target_size=self.target_size,
                use_clahe=True,
                use_gaussian=True,
                use_circular_mask=True
            )
            tensor = tensor.squeeze(0)  # Remove batch dim: (3, H, W)
        else:
            # Pipeline simples (só resize + normalização)
            if self.split == "train":
                transform = get_train_transforms(self.target_size)
            else:
                transform = get_val_transforms(self.target_size)
            tensor = transform(img_pil)
        
        # Augmentation geométrica (aplicada na imagem, não no tensor)
        if self.augment and self.split == "train":
            tensor = self.augment(tensor)
        
        return tensor, torch.tensor(label, dtype=torch.float32)


# ──────────────────────────────────────────────
# 2. LOOP DE TREINO
# ──────────────────────────────────────────────

def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    """Executa uma época de treino."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # (batch,) → (batch, 1)
        
        optimizer.zero_grad()
        
        # Mixed precision training (mais rápido em GPU)
        if scaler is not None:
            with torch.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Métricas
        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {running_loss/(batch_idx+1):.4f} | "
                  f"Acc: {correct/total:.4f}")
    
    return running_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    """Avalia o modelo no conjunto de validação."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    # Calcula métricas adicionais
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)  # Sensibilidade (importante em medicina!)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        "loss": running_loss / len(dataloader),
        "accuracy": correct / total,
        "precision": precision,
        "recall": recall,  # = sensibilidade
        "f1": f1
    }


# ──────────────────────────────────────────────
# 3. LOOP PRINCIPAL DE TREINO
# ──────────────────────────────────────────────

def train(args):
    # Configuração do device
    if torch.cuda.is_available():
        device = "cuda"
        scaler = torch.cuda.amp.GradScaler()  # mixed precision
        print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        scaler = None
        print("Usando Apple Silicon MPS")
    else:
        device = "cpu"
        scaler = None
        print("GPU não disponível, usando CPU (treino será lento)")
    
    # Datasets e DataLoaders
    print("\n📂 Carregando dataset...")
    train_dataset = RetinaDataset(
        args.data_dir, split="train",
        use_cv_preprocessing=args.use_cv_preprocessing
    )
    val_dataset = RetinaDataset(
        args.data_dir, split="val",
        use_cv_preprocessing=args.use_cv_preprocessing
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    
    # Modelo
    print("\nInicializando modelo...")
    model = RetinopatiaModel(pretrained=True, freeze_backbone=True)
    model = model.to(device)
    
    trainable = model.get_trainable_params()
    total = model.get_total_params()
    print(f"  Parâmetros treináveis: {trainable:,} / {total:,} "
          f"({100*trainable/total:.1f}%)")
    
    # Descongelar backbone após algumas épocas (fine-tuning progressivo)
    criterion = get_loss_function().to(device)
    optimizer = get_optimizer(model, lr=args.lr)
    
    # Scheduler: reduz LR quando val_loss para de melhorar
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Treino
    print(f"\n🚀 Iniciando treino: {args.epochs} épocas\n")
    best_model_weights = copy.deepcopy(model.state_dict())
    best_val_f1 = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"{'='*50}")
        print(f"Época {epoch}/{args.epochs}")
        
        # Descongelar backbone na metade do treino -> comentado para evitar overfitting em dataset pequeno
        # if epoch == args.epochs // 2:
        #     print("  → Descongelando backbone para fine-tuning completo...")
        #     for param in model.parameters():
        #         param.requires_grad = True
        #     optimizer = get_optimizer(model, lr=args.lr * 0.1)
        
        # Treino
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        
        # Validação
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Scheduler
        scheduler.step(val_metrics["loss"])
        
        elapsed = time.time() - t0
        print(f"\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_metrics['loss']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Recall: {val_metrics['recall']:.4f} | "
              f"F1: {val_metrics['f1']:.4f}")
        print(f"  Tempo: {elapsed:.1f}s")
        
        # Salva o melhor modelo (por F1, não só accuracy)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save({
                "epoch": epoch,
                "model_state_dict": best_model_weights,
                "val_acc": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "val_recall": val_metrics["recall"],
                "args": vars(args)
            }, args.save_path)
            print(f"  💾 Melhor modelo salvo! (F1: {best_val_f1:.4f})")
        
        # Histórico
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
    
    print(f"\n✅ Treino concluído! Melhor F1: {best_val_f1:.4f}")
    print(f"   Modelo salvo em: {args.save_path}")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treino - Retinopatia Diabética")
    parser.add_argument("--data_dir", type=str, default="./data/processed",
                        help="Diretório do dataset (com train/ e val/)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_path", type=str, default="modelo_retina.pth")
    parser.add_argument("--use_cv_preprocessing", action="store_true", default=True,
                        help="Aplica pipeline de pré-processamento (CLAHE, Gaussian, etc.)")
    
    args = parser.parse_args()
    train(args)
