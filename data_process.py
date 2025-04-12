import os
import cv2
import numpy as np
from pathlib import Path
#import kagglehub

'''os.environ["KAGGLEHUB_CACHE"] = r"C:/Users/User/Desktop/dc/cartoonify/data"
path = kagglehub.dataset_download("defileroff/comic-faces-paired-synthetic")
print("Path to dataset files:", path)
'''

def preprocess_images(input_dir, target_dir, output_dir, target_size=256):
    os.makedirs(output_dir, exist_ok=True)

    input_files = sorted(Path(input_dir).glob("*.jpg")) + sorted(Path(input_dir).glob("*.png"))
    target_files = sorted(Path(target_dir).glob("*.jpg")) + sorted(Path(target_dir).glob("*.png"))

    print(f"Found {len(input_files)} input files and {len(target_files)} target files.")

    if len(input_files) != len(target_files):
        raise ValueError("Mismatch between number of input and target images!")

    for idx, (input_file, target_file) in enumerate(zip(input_files, target_files)):
        input_img = cv2.imread(str(input_file))
        if input_img is None:
            print(f"Error reading input image: {input_file}")
            continue
        
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (target_size, target_size))

        target_img = cv2.imread(str(target_file))
        if target_img is None:
            print(f"Error reading target image: {target_file}")
            continue
        
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        target_img = cv2.resize(target_img, (target_size, target_size))

        # Save paired data as .npz file
        np.savez_compressed(
            os.path.join(output_dir, f"pair_{idx:05d}.npz"),
            input=input_img,
            target=target_img
        )

if __name__ == "__main__":
    try:
        preprocess_images(
            input_dir=r"/kaggle/input/comic-faces-paired-synthetic/face2comics_v1.0.0_by_Sxela/face2comics_v1.0.0_by_Sxela/face",
            target_dir=r"/kaggle/input/comic-faces-paired-synthetic/face2comics_v1.0.0_by_Sxela/face2comics_v1.0.0_by_Sxela/comics",
            output_dir=r"kaggle/working/processed_data_512",

            target_size=512
        )
    except Exception as e:
        print(f"Error: {e}")
