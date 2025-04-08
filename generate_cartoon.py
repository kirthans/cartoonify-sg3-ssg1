import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from generator import Generator

img_dim=256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = Generator(img_channels = 3, features = 64).to(device)
gen.load_state_dict(torch.load(r"C:/Users/Rama/Desktop/dc/cartoonify/working/checkpoints/generator.pth", map_location=device))
gen.eval()

transform = transforms.Compose([
            transforms.Resize((img_dim, img_dim)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
input_img= Image.open(r"C:/Users/Rama/Desktop/dc/cartoonify/h.png").convert("RGB")
img_tensor = transform(input_img).unsqueeze(0).to(device)
with torch.no_grad():
    output_tensor = gen(img_tensor)

output_tensor = (output_tensor * 0.5 + 0.5) * 255
output_image = output_tensor.squeeze().permute(1, 2, 0).cpu().numpy().astype("uint8")

output_pil = Image.fromarray(output_image)
output_pil.save(r"C:/Users/Rama/Desktop/dc/cartoonify/h_cartoon.jpg")