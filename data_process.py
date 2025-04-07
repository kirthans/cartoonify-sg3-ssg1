import torch
import torchvision
import cv2

img_path = '/kaggle/input/data/'
def preprocess_image(img_path, target_size=128):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.CN_COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size, target_size))
    return img
