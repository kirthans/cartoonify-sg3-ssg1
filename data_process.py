import os
import cv2
import numpy as np
from pathlib import Path

def preprocess_images(input_dir, target_dir, output_dir, target_size=128):
    os.makedirs(output_dir, exist_ok=True)

    input_files = sorted(Path(input_dir).glob("*.jpg")) + sorted(Path(input_dir).glob("*.png"))
    target_files = sorted(Path(target_dir).glob("*.jpg")) + sorted(Path(target_dir).glob("*.png"))

    if len(input_files) != len(target_files):
        raise ValueError("Mismatch between number of input and target images!")

    for idx, (input_file, target_file) in enumerate(zip(input_files, target_files)):
   
        input_img = cv2.imread(str(input_file))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (target_size, target_size))

        target_img = cv2.imread(str(target_file))
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        target_img = cv2.resize(target_img, (target_size, target_size))

        np.savez_compressed(
            os.path.join(output_dir, f"pair_{idx:05d}.npz"),
            input=input_img,
            target=target_img
        )

if __name__ == "__main__":
    preprocess_images(
        input_dir="/kaggle/input/comic-faces-paired-synthetic/input",
        target_dir="/kaggle/input/comic-faces-paired-synthetic/target",
        output_dir="/kaggle/working/data/processed",
        target_size=128  # Resize all images to 128x128
    )
