"""
prepare_mbrset.py
=================
Prepara o dataset mBRSET a partir do CSV já processado.

Estrutura esperada ANTES de rodar:
    mbrset/
    ├── images/          ← todas as imagens (1.1.jpg, 1.2.jpg, ...)
    └── labels.csv       ← CSV com colunas: image, label

Valores esperados na coluna 'label':
    - "retinopathy"  → pasta retinopatia/
    - "normal"       → pasta normal/

Estrutura gerada DEPOIS:
    dataset/
    ├── train/
    │   ├── normal/
    │   └── retinopatia/
    └── val/
        ├── normal/
        └── retinopatia/

Uso:
    python prepare_mbrset.py
    python prepare_mbrset.py --mbrset_dir ./mbrset --images_dir ./mbrset/images --output_dir ./dataset --val_split 0.2
"""

import os
import shutil
import argparse
import random
import csv
from collections import Counter


def prepare_mbrset(output_dir: str,
                   val_split: float = 0.2, seed: int = 42):
    """
    Organiza o mBRSET em pastas train/val com subpastas normal/retinopatia.

    Args:
        output_dir:  pasta de destino do dataset organizado
        val_split:   fração para validação (padrão 20%)
        seed:        semente aleatória para reprodutibilidade
    """
    random.seed(seed)

    mbrset_dir = 'data\mbrset'
    csv_path = os.path.join(mbrset_dir, "labels.csv")
    images_dir = os.path.join(mbrset_dir, "images")
    print(f"✓ CSV encontrado: {csv_path}")
    print(f"✓ Pasta de imagens: {images_dir}")

    # ── Lê o CSV ──────────────────────────────────────────────
    # Mapeia os valores do CSV para os nomes de pasta do projeto
    LABEL_MAP = {
        "retinopathy": "retinopatia",
        "retinopatia": "retinopatia",
        "normal":      "normal",
        "0":           "normal",
        "1":           "retinopatia",
    }

    samples = []   # lista de (caminho_absoluto, label_pasta)
    skipped = []   # imagens do CSV não encontradas em disco

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Detecta os nomes das colunas de forma flexível
        fieldnames = reader.fieldnames
        img_col   = next((c for c in fieldnames if c.lower() in ("image", "filename", "file", "img")), fieldnames[0])
        label_col = next((c for c in fieldnames if c.lower() in ("label", "class", "diagnosis", "target")), fieldnames[1])

        print(f"  Coluna de imagem: '{img_col}' | Coluna de label: '{label_col}'")

        for row in reader:
            img_name = row[img_col].strip()
            raw_label = row[label_col].strip().lower()

            label = LABEL_MAP.get(raw_label)
            if label is None:
                print(f"  ⚠ Label desconhecido ignorado: '{raw_label}' ({img_name})")
                continue

            img_path = os.path.join(images_dir, img_name)
            if os.path.exists(img_path):
                samples.append((img_path, label))
            else:
                skipped.append(img_name)

    if not samples:
        print("❌ Nenhuma imagem encontrada. Verifique o caminho das imagens.")
        return

    if skipped:
        print(f"\n⚠ {len(skipped)} imagens do CSV não foram encontradas em disco:")
        for s in skipped[:10]:
            print(f"   - {s}")
        if len(skipped) > 10:
            print(f"   ... e mais {len(skipped) - 10}")

    # ── Estatísticas ──────────────────────────────────────────
    label_counts = Counter(label for _, label in samples)
    total = len(samples)
    print(f"\n📊 Dataset mBRSET:")
    print(f"   Total de imagens:  {total}")
    print(f"   Normal:            {label_counts['normal']}  ({100*label_counts['normal']/total:.1f}%)")
    print(f"   Retinopatia:       {label_counts['retinopatia']}  ({100*label_counts['retinopatia']/total:.1f}%)")

    if label_counts['retinopatia'] > 0:
        pos_weight = label_counts['normal'] / label_counts['retinopatia']
        print(f"\n   💡 Dica: use pos_weight ≈ {pos_weight:.2f} no BCEWithLogitsLoss")
        print(f"      (em model.py, linha: pos_weight = torch.tensor([{pos_weight:.1f}]))")

    # ── Stratified split por paciente (baseado no prefixo do nome) ──
    # O mBRSET nomeia arquivos como "1.1.jpg", "1.2.jpg" (paciente.olho.jpg)
    # Agrupa por paciente para evitar data leakage (mesmo paciente em train e val)
    patients = {}
    for img_path, label in samples:
        fname = os.path.basename(img_path)
        # Extrai ID do paciente: "10.2.jpg" → "10"
        patient_id = fname.split(".")[0]
        if patient_id not in patients:
            patients[patient_id] = {"label": label, "images": []}
        patients[patient_id]["images"].append((img_path, label))

    # Separa pacientes por classe para stratified split
    patients_by_class = {"normal": [], "retinopatia": []}
    for pid, info in patients.items():
        patients_by_class[info["label"]].append(pid)

    for cls in patients_by_class:
        random.shuffle(patients_by_class[cls])

    # Divide pacientes em train/val (não imagens — evita leakage!)
    val_patients = set()
    for cls, pids in patients_by_class.items():
        n_val = max(1, int(len(pids) * val_split))
        val_patients.update(pids[:n_val])

    splits = {"train": [], "val": []}
    for img_path, label in samples:
        fname = os.path.basename(img_path)
        patient_id = fname.split(".")[0]
        split = "val" if patient_id in val_patients else "train"
        splits[split].append((img_path, label))

    print(f"\n   Separação (por paciente, sem data leakage):")
    print(f"   Val patients:   {len(val_patients)}")
    print(f"   Train patients: {len(patients) - len(val_patients)}")

    # ── Cria pastas e copia imagens ───────────────────────────
    for split_name, split_samples in splits.items():
        counts = Counter(label for _, label in split_samples)
        print(f"\n   {split_name.upper()}: {len(split_samples)} imagens "
              f"(normal: {counts['normal']}, retinopatia: {counts['retinopatia']})")

        for label in ["normal", "retinopatia"]:
            os.makedirs(os.path.join(output_dir, split_name, label), exist_ok=True)

        for img_path, label in split_samples:
            dest = os.path.join(output_dir, split_name, label, os.path.basename(img_path))
            if not os.path.exists(dest):
                shutil.copy2(img_path, dest)

    # ── Resumo final ──────────────────────────────────────────
    train_counts = Counter(l for _, l in splits["train"])
    val_counts   = Counter(l for _, l in splits["val"])

    print(f"""
✅ Dataset preparado em: {output_dir}/

   {output_dir}/
   ├── train/
   │   ├── normal/       ({train_counts['normal']} imgs)
   │   └── retinopatia/  ({train_counts['retinopatia']} imgs)
   └── val/
       ├── normal/       ({val_counts['normal']} imgs)
       └── retinopatia/  ({val_counts['retinopatia']} imgs)

# ▶ Próximo passo:
#    python train.py --data_dir {output_dir} --epochs 20 --batch_size 32
# """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara dataset mBRSET")
    parser.add_argument("--mbrset_dir",  type=str, default="./mbrset",
                        help="Pasta com o arquivo labels.csv")
    parser.add_argument("--images_dir",  type=str, default="./mbrset/images",
                        help="Pasta com todas as imagens .jpg")
    parser.add_argument("--output_dir",  type=str, default="./dataset",
                        help="Pasta de destino do dataset organizado")
    parser.add_argument("--val_split",   type=float, default=0.2,
                        help="Fração para validação (padrão: 0.2 = 20%%)")
    parser.add_argument("--seed",        type=int, default=42,
                        help="Semente aleatória para reprodutibilidade")

    args = parser.parse_args()
    prepare_mbrset(
                   args.output_dir, args.val_split, args.seed)