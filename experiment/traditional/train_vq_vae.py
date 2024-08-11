import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from VQVAE import VQVAE

# Parameters for STL-10
batch_size = 64
in_channels = 3
height = 96
width = 96
codebook_size = 512
dim = 128
beta = 0.25
epochs = 10
learning_rate = 1e-3

down_channels = [(in_channels, 32, 1), (32, 64, 2), (64, dim, 2)]
up_channels = [(dim, 64, 2), (64, 32, 2), (32, in_channels, 1)]

# Load STL-10 dataset
transform = transforms.Compose([
    transforms.Resize((height, width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.STL10(root='../data/stl10', split='train', download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_dataset = datasets.STL10(root='../data/stl10', split='test', download=True, transform=transform)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
output_dir = './results/stl10'

# Create a VQVAE instance
vqvae = VQVAE(down_channels, up_channels, codebook_size, dim, beta)

# Fit the VQVAE model
vqvae.fit(train_loader, eval_loader, epochs, learning_rate, device='cuda', save_dir=output_dir)

# Save the model
torch.save(vqvae.state_dict(), f'{output_dir}/vqvae_stl10.pth')
