import os, json, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from modules import DownBlock, VectorQuantizer, UpBlock
from utils.ssim import ssim
from utils.psnr import psnr
import lpips
import warnings
warnings.filterwarnings('ignore')

class VQVAE(nn.Module):
    def __init__(self, down_channels, up_channels, codebook_size, dim, beta=0.25):
        super(VQVAE, self).__init__()

        # Create encoder from DownBlocks
        self.encoder = nn.Sequential(
            *[DownBlock(in_ch, out_ch, scale_factor) for in_ch, out_ch, scale_factor in down_channels],
        )
        
        # Vector quantizer
        self.vector_quantizer = VectorQuantizer(codebook_size, dim, beta)
        
        # Create decoder from UpBlocks
        self.decoder = nn.Sequential(
            *[UpBlock(in_ch, out_ch, scale_factor) for in_ch, out_ch, scale_factor in up_channels]
        )

    def encode(self, x):
        z_e = self.encoder(x)
        return z_e
    
    def indices(self, x):
        indices = self.vector_quantizer.get_indices(x)
        return indices
    
    def quantize(self, x):
        quantized, _, _ , _, _ = self.vector_quantizer(x)
        return quantized
    
    def decode(self, x):
        x_recon = self.decoder(x)
        return x_recon

    def forward(self, x):
        z_e = self.encoder(x)
        quantized, codebook_loss, commitment_loss, perplexity_loss, loss = self.vector_quantizer(z_e)
        x_recon = self.decoder(quantized)
        return x_recon, codebook_loss, commitment_loss, perplexity_loss, loss

    def fit(self, train_dataloader, eval_dataloader, epochs, learning_rate=1e-3, device='cuda', save_dir='outputs'):
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, cooldown=1, threshold=0.01, verbose=True)
        lpips_network = lpips.LPIPS(net='alex').to(device)
        self.to(device)
        results = {}

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for epoch in range(epochs):
            train_loss, train_mse, train_codebook_loss, train_commitment_loss, train_perplexity_loss, train_psnr, train_ssim, train_lpips = self.process_epoch(train_dataloader, optimizer, device, train=True, lpips_network=lpips_network)
            eval_loss, eval_mse, eval_codebook_loss, eval_commitment_loss, eval_perplexity_loss, eval_psnr, eval_ssim, eval_lpips = self.process_epoch(eval_dataloader, optimizer, device, train=False, save_images=True, epoch=epoch, save_dir=save_dir, lpips_network=lpips_network)
            current_lr = optimizer.param_groups[0]['lr']
            results[epoch] = {
                'train': {'loss': train_loss, 'mse_loss': train_mse, 'codebook_loss': train_codebook_loss, 'commitment_loss': train_commitment_loss, 'perplexity_loss': train_perplexity_loss, 'psnr': train_psnr, 'ssim': train_ssim, "lpips": train_lpips},
                'eval': {'loss': eval_loss, 'mse_loss': eval_mse, 'codebook_loss': eval_codebook_loss, 'commitment_loss': eval_commitment_loss, 'perplexity_loss': eval_perplexity_loss, 'psnr': eval_psnr, 'ssim': eval_ssim, "lpips": eval_lpips},
                'lr' : current_lr
            }

            scheduler.step(eval_loss)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.3f}, MSE: {train_mse:.3f}, Codebook Loss: {train_codebook_loss:.3f}, Commitment Loss: {train_commitment_loss:.3f}, Perplexity Loss: {train_perplexity_loss:.3f}, PSNR: {train_psnr:.3f}, SSIM: {train_ssim:.3f}, LPIPS: {train_lpips:.3f}")
            print(f"Eval Loss: {eval_loss:.3f}, MSE: {eval_mse:.3f}, Codebook Loss: {eval_codebook_loss:.3f}, Commitment Loss: {eval_commitment_loss:.3f}, Perplexity Loss: {eval_perplexity_loss:.3f}, PSNR: {eval_psnr:.3f}, SSIM: {eval_ssim:.3f}, LPIPS: {eval_lpips:.3f}")
            print(f"Current LR: {current_lr}")
            
            # Save results and model
            with open(os.path.join(save_dir, f'results_epoch_{epoch+1}.json'), 'w') as fp:
                json.dump(results[epoch], fp, indent=4)
            
            torch.save(self.state_dict(), os.path.join(save_dir, f'vqvae_epoch_{epoch+1}.pth'))

    def process_epoch(self, dataloader, optimizer, device, train=True, save_images=False, epoch=None, save_dir='./', lpips_network=None):
        if train:
            self.train()
        else:
            self.eval()

        total_loss = total_mse = total_codebook = total_commitment = total_perplexity = total_psnr = total_ssim = total_lpips = count = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training" if train else "Evaluating")):
            inputs = batch[0].to(device)
            if train:
                optimizer.zero_grad()
            with torch.set_grad_enabled(train):
                x_recon, codebook_loss, commitment_loss, perplexity_loss, loss = self.forward(inputs)

                # Loss
                mse_loss = nn.functional.mse_loss(x_recon, inputs)
                psnr_loss = (1.0 - (psnr((inputs * 0.5 + 0.5), (x_recon * 0.5 + 0.5))/48))
                ssim_loss = (1.0 - ssim((inputs * 0.5 + 0.5), (x_recon * 0.5 + 0.5)))
                lpips_loss = lpips_network(inputs, x_recon).mean()
                
                # Totals
                total_mse += mse_loss.item()
                total_codebook += codebook_loss.item()
                total_commitment += commitment_loss.item()
                total_perplexity += perplexity_loss.item()
                total_psnr += psnr_loss.item()
                total_ssim += ssim_loss.item()
                total_lpips += lpips_loss.item()
                total_loss += (mse_loss.item() + loss.item() + psnr_loss.item() + ssim_loss.item() + lpips_loss.item())

                
                if batch_idx == 0 and save_images and not train:  # Save images for the first batch of evaluation only
                    with torch.set_grad_enabled(False):
                        self.save_images(inputs, x_recon, batch_idx, epoch, save_dir, lpips_network)

                if train:
                    (mse_loss + loss + ((psnr_loss + ssim_loss + lpips_loss)*0.1)).backward()
                    optimizer.step()

            count += 1

        avg_loss = total_loss / count
        avg_mse = total_mse / count
        avg_codebook = total_codebook / count
        avg_commitment = total_commitment / count
        avg_perplexity = total_perplexity / count
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        avg_lpips = total_lpips / count

        return avg_loss, avg_mse, avg_codebook, avg_commitment, avg_perplexity, avg_psnr, avg_ssim, avg_lpips
    
    def save_images(self, inputs, reconstructions, batch_idx, epoch, save_dir, lpips_network):
        fig, axes = plt.subplots(2, 10, figsize=(15, 6))

        for i in range(min(10, inputs.shape[0])):
            original = inputs[i]
            reconstruction = reconstructions[i]
            mse = F.mse_loss(reconstruction, original).item()
            p_score = psnr((original * 0.5 + 0.5).unsqueeze(0), (reconstruction * 0.5 + 0.5).unsqueeze(0))
            s_score = ssim((original * 0.5 + 0.5).unsqueeze(0), (reconstruction * 0.5 + 0.5).unsqueeze(0))
            l_score = lpips_network((original * 0.5 + 0.5).unsqueeze(0), (reconstruction * 0.5 + 0.5).unsqueeze(0)).mean()

            # Display original images in the top row
            axes[0, i].imshow((original.permute(1, 2, 0) * 0.5 + 0.5).cpu().numpy())
            axes[0, i].axis('off')  # Hide axes

            # Display reconstructed images in the bottom row
            axes[1, i].imshow((reconstruction.permute(1, 2, 0) * 0.5 + 0.5).cpu().numpy())
            axes[1, i].axis('off')  # Hide axes

            # Add centered text
            axes[1, i].text(0.5, -0.5, f'MSE: {mse:.3f}\nPSNR: {p_score:.3f}\nSSIM: {s_score:.3f}\nLPIPS: {l_score:.3f}',
                            horizontalalignment='center', verticalalignment='center',
                            transform=axes[1, i].transAxes, fontname='Consolas', fontsize=14)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/comparison_epoch_{epoch+1}_batch_{batch_idx}.png")
        plt.close()