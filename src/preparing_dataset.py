import os
import shutil
import pandas as pd
from tqdm import tqdm #progress bar

BASE_PATH = "data/raw"
OUTPUT_PATH = "data/processed"

train_csv = os.path.join(BASE_PATH, "train_1.csv")
val_csv = os.path.join(BASE_PATH, "valid.csv")

train_images_dir = os.path.join(BASE_PATH, "train_images", "train_images")
val_images_dir = os.path.join(BASE_PATH, "val_images", "val_images")

# APTOS 2019 BD samples are divided into five categories: no Diabetic Retinopathy (DR) -> 0, others above 0 -> mild DR, moderate DR, severe DR, and proliferative DR
def organize_split(csv_path, images_dir, split_name):
    df = pd.read_csv(csv_path)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_id = row["id_code"]
        label = row["diagnosis"]

        if label == 0:
            class_name = "normal"
        else:
            class_name = "retinopatia"

        src = os.path.join(images_dir, img_id + ".png")
        dst_dir = os.path.join(OUTPUT_PATH, split_name, class_name)
        os.makedirs(dst_dir, exist_ok=True)

        dst = os.path.join(dst_dir, img_id + ".png")

        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"Imagem nao encontrada: {src}")

organize_split(train_csv, train_images_dir, "train")
organize_split(val_csv, val_images_dir, "val")