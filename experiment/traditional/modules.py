import torch
import torch.nn as nn
import torch.nn.functional as F

# DownBlock as described by the manuscript
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=4 if scale_factor > 1 else 3, stride=scale_factor, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4 if scale_factor > 1 else 3, stride=scale_factor, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.activation = nn.Tanh()

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x + residual)
        return x

# UpBlock as described by the manuscript
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpBlock, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4 if scale_factor > 1 else 3, stride=scale_factor, padding=1)
        self.deconv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.residual = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4 if scale_factor > 1 else 3, stride=scale_factor, padding=1),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.activation = nn.Tanh()

    def forward(self, x):
        residual = self.residual(x)
        x = self.deconv1(x)
        x = self.activation(x)
        x = self.deconv2(x)
        x = self.activation(x + residual)
        return x

# VectorQuantizer, as described in the manuscript, has added perplexity loss for codes.
class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, dim, beta=0.25):
        super(VectorQuantizer, self).__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.beta = beta

        # Initialize the codebook as an nn.Embedding
        torch.manual_seed(5262024)
        self.codebook = nn.Embedding(codebook_size, dim)
        self.codebook.weight.data.uniform_(-1/codebook_size, 1/codebook_size)

    def forward(self, x):
        # Get the input shape
        B, C, W, H = x.shape
        assert C == self.dim, "The channel dimension must match the codebook embedding dimension"

        # Flatten the input except for the batch dimension
        flat_x = x.permute(0, 2, 3, 1).contiguous().view(-1, self.dim)

        # Calculate distances between input and codebook vectors
        distances = (flat_x ** 2).sum(1, keepdim=True) + (self.codebook.weight ** 2).sum(1) - 2 * flat_x @ self.codebook.weight.t()

        # Find the nearest codebook vectors
        _, indices = torch.min(distances, dim=1)
        quantized = self.codebook(indices)

        # Perplexity calculation using log-sum-exp for stability
        unique_indices, counts = indices.unique(return_counts=True)
        log_probs = torch.log(counts.float()) - torch.log(torch.tensor(indices.numel(), dtype=torch.float))
        entropy = -torch.sum(torch.exp(log_probs) * log_probs)  # Direct calculation of entropy from log probabilities
        perplexity = torch.exp(entropy)  # Perplexity as exp(entropy)

        # Minimizing the reciprocal of perplexity to promote higher values
        perplexity_loss = 1 / perplexity

        # Reshape quantized to the original input shape
        quantized = quantized.view(B, W, H, C).permute(0, 3, 1, 2).contiguous()

        # Calculate losses
        commitment_loss = F.mse_loss(quantized.detach(), x)
        codebook_loss = F.mse_loss(quantized, x.detach())
        loss = commitment_loss + (self.beta * codebook_loss) + (self.beta * perplexity_loss)

        # Straight-through estimator for the gradient
        quantized = x + (quantized - x).detach()

        return quantized, codebook_loss, commitment_loss, perplexity_loss, loss
    
    def get_indices(self, x):
        # Get the input shape
        B, C, W, H = x.shape
        assert C == self.dim, "The channel dimension must match the codebook embedding dimension"

        # Flatten the input except for the batch dimension
        flat_x = x.permute(0, 2, 3, 1).contiguous().view(-1, self.dim)

        # Calculate distances between input and codebook vectors
        distances = (flat_x ** 2).sum(1, keepdim=True) + (self.codebook.weight ** 2).sum(1) - 2 * flat_x @ self.codebook.weight.t()

        # Find the nearest codebook vectors
        _, indices = torch.min(distances, dim=1)

        return indices.view(B, W * H)
