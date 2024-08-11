import torch, torchhd, os, json
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from HDVQVAE import HDVQVAE
from tqdm import tqdm

def sign(tensor):
    return torch.where(tensor < 0, torch.tensor(-1.0), torch.tensor(1.0))

# Parameters for CIFAR-10
batch_size = 64
in_channels = 3
height = 96
width = 96
codebook_size = 512
dim = 128
beta = 0.25
epochs = 10
learning_rate = 0.1
lr_drop = 0.95

down_channels = [(in_channels, 32, 1), (32, 64, 2), (64, dim, 2)]
up_channels = [(dim, 64, 2), (64, 32, 2), (32, in_channels, 1)]

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize((height, width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.STL10(root='../data/stl10', split='train', download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_dataset = datasets.STL10(root='../data/stl10', split='test', download=True, transform=transform)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

# Latent space size
latent_space_dim = 24 * 24 * dim

# Path to the saved model
model_path = './results/stl10/vqvae_stl10.pth'

# Create a VQVAE instance
vqvae = HDVQVAE(down_channels, up_channels, codebook_size, dim, beta)

# Load the trained model
vqvae.load_state_dict(torch.load(model_path))

# Ensure the model is in evaluation mode
vqvae.eval()
vqvae.to("cuda")

class_matrix = torchhd.ensure_vsa_tensor(torch.zeros((10, latent_space_dim))).to("cuda")
positional_vector = torchhd.random(1, latent_space_dim).to("cuda")

dir = "./results/hdc_stl_classifier"
if not os.path.exists(dir):
     os.makedirs(dir)

# Process Training Data
for batch_idx, batch in enumerate(tqdm(train_loader, desc="PreTraining")):
    inputs = batch[0].to("cuda")
    targets = batch[1].to("cuda")
    latents = vqvae.quantize(vqvae.encode(inputs))

    hd_latents = latents.permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
    if hd_latents[0].size(0) != (latent_space_dim):
        break
    hd_latents = torchhd.bind(hd_latents, positional_vector)
    
    for i in range(hd_latents.size(0)):
        class_matrix[targets[i]] += hd_latents[i]
    
    class_matrix = sign(class_matrix)

correct = 0
total = 0
# Process Eval Data
for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluation")):
    inputs = batch[0].to("cuda")
    targets = batch[1].to("cuda")
    latents = vqvae.quantize(vqvae.encode(inputs))

    hd_latents = latents.permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
    if hd_latents[0].size(0) != (latent_space_dim):
        break
    hd_latents = torchhd.bind(hd_latents, positional_vector)
    
    for i in range(hd_latents.size(0)):
        total += 1
        if torch.argmax(torchhd.cosine_similarity(hd_latents[i],class_matrix)).item() == targets[i].item():
            correct += 1

print(f"Accuracy: {correct/total}")

# Process Training Data
for epoch in range(epochs):
    correct = 0
    total = 0
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        inputs = batch[0].to("cuda")
        targets = batch[1].to("cuda")
        latents = vqvae.quantize(vqvae.encode(inputs))

        hd_latents = latents.permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
        if hd_latents[0].size(0) != (latent_space_dim):
            break
        hd_latents = torchhd.bind(hd_latents, positional_vector)
        
        for i in range(hd_latents.size(0)):
            total += 1
            pred = torch.argmax(torchhd.cosine_similarity(hd_latents[i],class_matrix)).item()
            if pred == targets[i].item():
                correct += 1
            else:
                class_matrix[targets[i]] += hd_latents[i] * learning_rate
                class_matrix[pred] -= hd_latents[i] * learning_rate
    class_matrix = sign(class_matrix)
    learning_rate = learning_rate * lr_drop
    print(f"Train Accuracy: {correct/total} Learning Rate: {learning_rate}")
    training_accuracy = correct/total

    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training Eval")):
        inputs = batch[0].to("cuda")
        targets = batch[1].to("cuda")
        latents = vqvae.quantize(vqvae.encode(inputs))

        hd_latents = latents.permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
        if hd_latents[0].size(0) != (latent_space_dim):
            break
        hd_latents = torchhd.bind(hd_latents, positional_vector)
        
        for i in range(hd_latents.size(0)):
            total += 1
            pred = torch.argmax(torchhd.cosine_similarity(hd_latents[i],class_matrix)).item()
            if pred == targets[i].item():
                correct += 1

    print(f"Train Eval Accuracy: {correct/total} Learning Rate: {learning_rate}")
    training_accuracy = correct/total

    correct = 0
    total = 0
    # Process Eval Data
    for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluation")):
        inputs = batch[0].to("cuda")
        targets = batch[1].to("cuda")
        latents = vqvae.quantize(vqvae.encode(inputs))

        hd_latents = latents.permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
        if hd_latents[0].size(0) != (latent_space_dim):
            break
        hd_latents = torchhd.bind(hd_latents, positional_vector)
        
        for i in range(hd_latents.size(0)):
            total += 1
            if torch.argmax(torchhd.cosine_similarity(hd_latents[i],class_matrix)).item() == targets[i].item():
                correct += 1
    print(f"Eval Accuracy: {correct/total}")
    eval_accuracy = correct/total

    data = {
        "training_accuracy" : training_accuracy,
        "eval_accuracy": eval_accuracy,
        "lr" : learning_rate
    }
    with open(os.path.join(dir, f'results_epoch_{epoch+1}.json'), 'w') as fp:
        json.dump(data, fp, indent=4)
