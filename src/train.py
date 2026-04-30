"""
train.py
Fine-tuning da ResNet18 para classificacao binaria de retinopatia diabetica.

Uso:
    python train.py --data_dir ./data/processed --epochs 20 --batch_size 16

Estrutura esperada do dataset:
    data/processed/
    ├── train/
    │   ├── normal/
    │   └── retinopatia/
    └── val/
        ├── normal/
        └── retinopatia/
"""

import os
import copy
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from model import RetinopatiaModel, get_optimizer
from preprocessing import get_train_transforms, get_val_transforms, preprocess_retina_image


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RetinaDataset(Dataset):
    """
    Dataset de imagens de retina com pre-processamento de visao computacional.

    Aplica o pipeline completo (CLAHE, mascara circular, etc.) em cada imagem
    antes de passa-la para o modelo. No split de treino, augmentation
    geometrica e de cor e aplicada antes do pre-processamento.
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
                print(f"Diretorio nao encontrado: {class_dir}")
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp')):
                    self.samples.append((os.path.join(class_dir, fname), label))

        self.augment = (split == "train")

        print(f"Dataset '{split}': {len(self.samples)} imagens "
              f"({sum(1 for _, l in self.samples if l == 0)} normal, "
              f"{sum(1 for _, l in self.samples if l == 1)} retinopatia)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img_pil = Image.open(img_path).convert("RGB")

        # Augmentation em PIL antes do pre-processamento para evitar
        # artefatos de aplicar transformacoes em tensores normalizados
        if self.augment:
            augment_pil = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=30),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])
            img_pil = augment_pil(img_pil)

        if self.use_cv_preprocessing:
            tensor = preprocess_retina_image(
                img_pil,
                target_size=self.target_size,
                use_clahe=True,
                use_gaussian=False,
                use_circular_mask=True
            )
            tensor = tensor.squeeze(0)
        else:
            transform = (get_train_transforms(self.target_size)
                         if self.split == "train"
                         else get_val_transforms(self.target_size))
            tensor = transform(img_pil)

        return tensor, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Loop de treino e avaliacao
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    """Executa uma epoca de treino e retorna loss e accuracy medios."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()

        if scaler is not None:
            # Mixed precision: reduz uso de memoria e acelera em GPU
            with torch.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        # Threshold 0.6 no treino e levemente mais conservador para
        # equilibrar precisao e recall durante o aprendizado inicial
        preds = (torch.sigmoid(outputs) >= 0.6).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {running_loss/(batch_idx+1):.4f} | "
                  f"Acc: {correct/total:.4f}")

    return running_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    """
    Avalia o modelo no conjunto de validacao.

    Calcula accuracy, precision, recall e F1. O recall e a metrica mais
    importante em diagnostico: um falso negativo (nao detectar retinopatia
    real) e mais grave do que um falso positivo.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()

            # Debug do primeiro batch para verificar se o modelo esta
            # gerando predicoes variadas (nao colapsando em uma so classe)
            if i == 0:
                print("Preds unicos:", preds.unique())
                print("Labels unicos:", labels.unique())
                print("Probabilidades:", probs[:10])

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "loss":      running_loss / len(dataloader),
        "accuracy":  correct / total,
        "precision": precision,
        "recall":    recall,
        "f1":        f1
    }


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------

def train(args):
    # Configuracao do device
    if torch.cuda.is_available():
        device = "cuda"
        scaler = torch.cuda.amp.GradScaler()
        print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        scaler = None
        print("Usando Apple Silicon MPS")
    else:
        device = "cpu"
        scaler = None
        print("GPU nao disponivel, usando CPU (treino sera lento)")

    print("\nCarregando dataset...")
    train_dataset = RetinaDataset(args.data_dir, split="train",
                                  use_cv_preprocessing=True)
    val_dataset   = RetinaDataset(args.data_dir, split="val",
                                  use_cv_preprocessing=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=(device == "cuda"))

    print("\nInicializando modelo...")
    model = RetinopatiaModel(pretrained=True, freeze_backbone=True)
    model = model.to(device)

    trainable = model.get_trainable_params()
    total     = model.get_total_params()
    print(f"  Parametros treinaveis: {trainable:,} / {total:,} "
          f"({100*trainable/total:.1f}%)")

    # pos_weight calculado da proporcao real do dataset para compensar
    # o desbalanceamento entre normal e retinopatia
    n_normal = sum(1 for _, l in train_dataset.samples if l == 0)
    n_retino = sum(1 for _, l in train_dataset.samples if l == 1)
    pw = n_normal / max(n_retino, 1)
    print(f"  pos_weight automatico: {pw:.2f} ({n_normal} normal / {n_retino} retinopatia)")

    pos_weight = torch.tensor([pw]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = get_optimizer(model, lr=args.lr)

    # Reduz o lr pela metade se o val_loss nao melhorar por 3 epocas
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    print(f"\nIniciando treino: {args.epochs} epocas\n")
    best_weights = copy.deepcopy(model.state_dict())
    best_val_f1  = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"{'='*50}")
        print(f"Epoca {epoch}/{args.epochs}")

        # Fine-tuning progressivo: descongela o backbone na metade do treino
        # com lr reduzido para nao destruir os pesos pre-treinados
        if epoch == max(2, args.epochs // 2):
            print("  Descongelando backbone para fine-tuning completo...")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = get_optimizer(model, lr=args.lr * 0.1)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler)

        val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_metrics["loss"])

        elapsed = time.time() - t0
        print(f"\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_metrics['loss']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Recall: {val_metrics['recall']:.4f} | "
              f"F1: {val_metrics['f1']:.4f}")
        print(f"  Tempo: {elapsed:.1f}s")

        # Salva o melhor modelo por F1 (mais confiavel que accuracy em
        # datasets desbalanceados)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_weights = copy.deepcopy(model.state_dict())
            torch.save({
                "epoch":            epoch,
                "model_state_dict": best_weights,
                "val_acc":          val_metrics["accuracy"],
                "val_f1":           val_metrics["f1"],
                "val_recall":       val_metrics["recall"],
                "args":             vars(args)
            }, args.save_path)
            print(f"  Melhor modelo salvo! (F1: {best_val_f1:.4f})")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])

    print(f"\nTreino concluido! Melhor F1: {best_val_f1:.4f}")
    print(f"Modelo salvo em: {args.save_path}")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treino - Retinopatia Diabetica")
    parser.add_argument("--data_dir",    type=str,   default="./data/processed")
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--save_path",   type=str,   default="modelo_retina.pth")
    parser.add_argument("--no_cv_preprocessing", action="store_false",
                        dest="use_cv_preprocessing",
                        help="Desliga o pre-processamento avancado")
    parser.set_defaults(use_cv_preprocessing=True)

    args = parser.parse_args()
    train(args)