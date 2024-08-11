import torch, os, json
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pandas as pd
from torch.utils.data import DataLoader
from VQVAE import VQVAE
from tqdm import tqdm

# Parameters for STL10
batch_size = 64
in_channels = 3
height = 96
width = 96
codebook_size = 512
dim = 128
beta = 0.25
epochs = 10
learning_rate = 1e-3
lr_drop = 0.95

down_channels = [(in_channels, 32, 1), (32, 64, 2), (64, dim, 2)]
up_channels = [(dim, 64, 2), (64, 32, 2), (32, in_channels, 1)]

# Load STL10 dataset
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
vqvae = VQVAE(down_channels, up_channels, codebook_size, dim, beta)

# Load the trained model
vqvae.load_state_dict(torch.load(model_path))

# Ensure the model is in evaluation mode
vqvae.eval()
vqvae.to("cuda")

def calculate_confusion_matrix(targets, predictions, num_classes):
    # Create a DataFrame from targets and predictions
    data = pd.DataFrame({'Targets': targets, 'Predictions': predictions})
    # Create a confusion matrix
    cm = pd.crosstab(data['Targets'], data['Predictions'], rownames=['Actual'], colnames=['Predicted'], dropna=False)
    # Ensure the confusion matrix has all possible classes
    for class_idx in range(num_classes):
        if class_idx not in cm:
            cm[class_idx] = 0
        if class_idx not in cm.index:
            cm.loc[class_idx] = 0
    cm = cm.sort_index().sort_index(axis=1)
    return cm

class Classifier(nn.Module):
    def __init__(self, vqvae : VQVAE) -> None:
        super(Classifier, self).__init__()
        self.vqvae = vqvae
        self.cnn = nn.Sequential(
            nn.Conv2d(128, 64, 4, 2, 1),
            nn.Tanh(),
            nn.Conv2d(64, 32, 4, 2, 1),
            nn.Tanh(),
            nn.Conv2d(32, 8, 4, 2, 1),
            nn.Flatten(),
            nn.Linear(72,32),
            nn.ReLU(),
            nn.Linear(32,10)
        )

    def forward(self, x):
        return self.cnn(x)
    
    def fit(self, train_dataloader, eval_dataloader, epochs = 10, learning_rate=1e-3, dir="./results/cnn_stl_classifier"):
        if not os.path.exists(dir):
            os.makedirs(dir)
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, cooldown=1, threshold=0.01, verbose=True)
        cross_loss = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            # Process Training Data
            total_loss = 0
            count = 0
            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training")):
                optimizer.zero_grad()
                inputs = batch[0].to("cuda")
                targets = batch[1].to("cuda")
                targets = nn.functional.one_hot(targets,10).float()
                latents = vqvae.quantize(vqvae.encode(inputs))

                features = self.forward(latents)
                class_loss = cross_loss(features,targets)

                loss = class_loss
                total_loss += loss
                loss.backward()
                optimizer.step()
                count+=1
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch: {epoch+1} | Avg Classification Loss: {(total_loss/count).item()} | Current LR: {current_lr}")
            scheduler.step(total_loss/count)
            my_loss = (total_loss/count).item()

            correct = 0
            total = 0
            all_targets = []
            all_predictions = []
            # Process Eval Data
            for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="Evaluation")):
                inputs = batch[0].to("cuda")
                targets = batch[1].to("cuda")
                latents = vqvae.quantize(vqvae.encode(inputs))

                features = self.forward(latents)
                features = nn.functional.softmax(features, dim=1)
                predictions = torch.argmax(features, dim=1)

                correct += torch.sum(predictions == targets).item()
                total += targets.size(0)
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

            print(f"Eval Accuracy: {correct/total}")
            eval_accuracy = correct/total

            cm = calculate_confusion_matrix(all_targets, all_predictions, 10)
            cm_filename = os.path.join(dir, f'confusion_matrix_eval_epoch_{epoch+1}.csv')
            cm.to_csv(cm_filename, index=True)

            correct = 0
            total = 0
            all_targets = []
            all_predictions = []
            # Process Eval Data
            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Evaluation on training data")):
                inputs = batch[0].to("cuda")
                targets = batch[1].to("cuda")
                latents = vqvae.quantize(vqvae.encode(inputs))

                features = self.forward(latents)
                features = nn.functional.softmax(features, dim=1)
                predictions = torch.argmax(features, dim=1)

                correct += torch.sum(predictions == targets).item()
                total += targets.size(0)
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

            print(f"Eval on training data Accuracy: {correct/total}")
            training_accuracy = correct/total

            cm = calculate_confusion_matrix(all_targets, all_predictions, 10)
            cm_filename = os.path.join(dir, f'confusion_matrix_train_epoch_{epoch+1}.csv')
            cm.to_csv(cm_filename, index=True)

            data = {
                "class cross entropy loss": my_loss,
                "training_accuracy" : training_accuracy,
                "eval_accuracy": eval_accuracy,
                "lr" : current_lr
            }
            with open(os.path.join(dir, f'results_epoch_{epoch+1}.json'), 'w') as fp:
                json.dump(data, fp, indent=4)

        torch.save(self.state_dict(), os.path.join(dir, f'cnn_classifier.pth'))
        
classifier = Classifier(vqvae).to("cuda")

classifier.fit(train_loader,eval_loader,epochs,learning_rate)
    
