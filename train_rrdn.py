import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F  # <-- required for resizing

from models.RRDN_arch import RRDN
# -----------------------------
# Dataset
# -----------------------------
class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, lr_size=(64, 64), hr_size=(256, 256)):
        super(SRDataset, self).__init__()
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_images = sorted(os.listdir(lr_dir))
        self.hr_images = sorted(os.listdir(hr_dir))

        # Resize and convert to tensor
        self.lr_transform = transforms.Compose([
            transforms.Resize(lr_size, interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])

        self.hr_transform = transforms.Compose([
            transforms.Resize(hr_size, interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])

        self.paired_files = [
            (os.path.join(self.lr_dir, lr), os.path.join(self.hr_dir, hr))
            for lr, hr in zip(self.lr_images, self.hr_images)
            if os.path.splitext(lr)[0] == os.path.splitext(hr)[0]
        ]
        assert len(self.paired_files) > 0, "No matching LR/HR image pairs found!"

    def __len__(self):
        return len(self.paired_files)

    def __getitem__(self, idx):
        lr_path, hr_path = self.paired_files[idx]
        lr = Image.open(lr_path).convert('RGB')
        hr = Image.open(hr_path).convert('RGB')

        lr = self.lr_transform(lr)
        hr = self.hr_transform(hr)

        return lr, hr
def train_rrdn_model():
    # Paths
    lr_dir = '.\dataset\LR'
    hr_dir = '.\dataset\HR'
    save_path = './models/rrdn_model.pth'

    # Hyperparameters
    batch_size = 4
    num_epochs = 100
    learning_rate = 1e-4
    scale = 4

    # Dataset & Dataloader
    dataset = SRDataset(lr_dir, hr_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RRDN(in_nc=3, out_nc=3, nf=64, nb=16, gc=32, scale=scale).to(device)

    # Optimizer & Loss
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for lr, hr in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            lr = lr.to(device)
            hr = hr.to(device)

            sr = model(lr)

            # Resize HR to match SR if sizes mismatch
            if sr.shape != hr.shape:
                hr = F.interpolate(hr, size=sr.shape[2:], mode='bicubic', align_corners=False)

            loss = criterion(sr, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss/len(dataloader):.4f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), save_path)
            print(f"\u2705 Model checkpoint saved at: {save_path}")

    # Final model
    torch.save(model.state_dict(), save_path)
    print(f"\n\u2705 Training complete. Final model saved at: {save_path}")


if __name__ == '__main__':
    train_rrdn_model()
