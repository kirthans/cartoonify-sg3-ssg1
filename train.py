import os
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.nn as nn
from torch import optim
from tqdm import tqdm


from generator import Generator
from discriminator import Discriminator

class Pix2PixTrain:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_directories()
        
        self.batch_size = 4
        self.img_dim = 512
        self.lr = 2e-4
        self.l1_lambda = 200
        self.checkpoint_interval = 100
        #self.num_epochs = 10
        
        self.init_models()
        self.init_data()
        self.writer = SummaryWriter(self.log_dir)

    def setup_directories(self):
        self.data_dir = Path("/kaggle/input/your-dataset-name")
        self.log_dir = Path("/kaggle/working/logs")
        self.ckpt_dir = Path("/kaggle/working/checkpoints")
        self.ckpt_dir.mkdir(exist_ok=True)

    def init_models(self):
        self.gen = Generator(img_channels=3).to(self.device)
        self.disc = Discriminator(img_channels=3).to(self.device)
        
        if (self.ckpt_dir/"generator.pth").exists():
            self.load_checkpoint()

    def init_data(self):
        transform = transforms.Compose([
            #transforms.Resize(self.img_dim),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        dataset = self.create_dataset(transform)
        self.loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def create_dataset(self):
        class PairedDataset(torch.utils.data.Dataset):
            def __init__(self, data_dir):
                self.files = sorted(Path(data_dir).glob("*.npz"))

            def __len__(self):
                return len(self.files)

            def __getitem__(self, idx):
                data = np.load(self.files[idx])
                
                # Convert HWC -> CHW and normalize to [-1, 1]
                input_img = torch.tensor(data['input']).permute(2, 0, 1).float() / 127.5 - 1
                target_img = torch.tensor(data['target']).permute(2, 0, 1).float() / 127.5 - 1
                
                return input_img, target_img

        return PairedDataset(self.data_dir)


    def train(self, num_epochs):
        opt_gen = optim.Adam(self.gen.parameters(), lr=self.lr, betas=(0.5, 0.999))
        opt_disc = optim.Adam(self.disc.parameters(), lr=self.lr, betas=(0.5, 0.999))
        bce, l1_loss = nn.BCEWithLogitsLoss(), nn.L1Loss()

        for epoch in range(num_epochs):
            loop = tqdm(self.loader, leave=True)
            for batch_idx, (input_img, target_img) in enumerate(loop):
                input_img, target_img = input_img.to(self.device), target_img.to(self.device)
                
                # Discriminator step
                fake_img = self.gen(input_img)
                disc_real = self.disc(input_img, target_img)
                disc_fake = self.disc(input_img, fake_img.detach())
                loss_disc = (bce(disc_real, torch.ones_like(disc_real)) + 
                            bce(disc_fake, torch.zeros_like(disc_fake))) / 2
                
                opt_disc.zero_grad()
                loss_disc.backward()
                opt_disc.step()

                # Generator step
                disc_fake = self.disc(input_img, fake_img)
                loss_gen = bce(disc_fake, torch.ones_like(disc_fake)) + l1_loss(fake_img, target_img)*self.l1_lambda
                
                opt_gen.zero_grad()
                loss_gen.backward()
                opt_gen.step()

                # Logging in tensorboard every 100 batcches(im using this to view loss)(port 6006)
                if batch_idx % 100 == 0:
                    self.log_tensorboard(input_img, target_img, fake_img, epoch, batch_idx)
                    
                if batch_idx % self.checkpoint_interval == 0 or batch_idx==len(self.load):
                    self.save_checkpoint()

    def log_tensorboard(self, input_img, target_img, fake_img, epoch, batch_idx):
        # Denormalize images
        input_img = (input_img * 0.5 + 0.5)
        fake_img = (fake_img * 0.5 + 0.5)
        target_img = (target_img * 0.5 + 0.5)
        
        input_grid = make_grid(input_img[:4], nrow=2, normalize=True)
        fake_grid = make_grid(fake_img[:4], nrow=2, normalize=True)
        target_grid = make_grid(target_img[:4], nrow=2, normalize=True)
        
        # TensorBoard(the actual images once every 100 batches)
        self.writer.add_image("Input", input_grid, global_step=epoch*len(self.loader)+batch_idx)
        self.writer.add_image("Fake", fake_grid, global_step=epoch*len(self.loader)+batch_idx)
        self.writer.add_image("Target", target_grid, global_step=epoch*len(self.loader)+batch_idx)

    def save_checkpoint(self):
        torch.save(self.gen.state_dict(), self.ckpt_dir/"generator.pth")
        torch.save(self.disc.state_dict(), self.ckpt_dir/"discriminator.pth")

    def load_checkpoint(self):
        self.gen.load_state_dict(torch.load(self.ckpt_dir/"generator.pth"))
        self.disc.load_state_dict(torch.load(self.ckpt_dir/"discriminator.pth"))

'''if __name__ == "__main__":
    trainer = Pix2PixTrain()
    trainer.train(num_epochs=10)
'''
