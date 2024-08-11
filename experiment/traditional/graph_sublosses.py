import os
import json
import matplotlib.pyplot as plt
import glob
import re

# Font settings for matplotlib
plt.rcParams.update({'font.size': 14, 'font.family': 'Consolas'})

# Function to load JSON data from a file
def load_json_data(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

# Extract epoch number from filename
def extract_epoch(filename):
    # Assuming the filename format ends with a number like "data_10.json"
    match = re.search(r'(\d+)\.json$', filename)
    if match:
        return int(match.group(1))
    return 0  # Default to 0 if no number is found

# Path to the results directory
results_dir = './results'

# Collect all subfolder paths within the results directory
subfolders = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, f))]
subfolders = [msf for msf in subfolders if not "hdc" in msf and not "cnn" in msf and not "mlp" in msf]

# Process each subfolder
for subfolder in subfolders:
    # Data structure to hold all the metrics from all JSONs
    metrics = {
        'train_loss': [],
        'eval_loss': [],
        'mse_loss': [],
        'commitment_loss': [],
        'psnr': [],
        'ssim': [],
        'lpips': []
    }

    # Find all JSON files in the current subfolder and sort them based on the epoch number
    json_files = sorted(glob.glob(os.path.join(subfolder, '*.json')), key=extract_epoch)

    # Epochs list to store the epoch numbers
    epochs = []

    # Load each JSON file
    for json_file in json_files:
        epoch = extract_epoch(json_file)
        epochs.append(epoch)
        data = load_json_data(json_file)
        # Append the loss values to the respective lists
        metrics['train_loss'].append(data['train']['loss'])
        metrics['eval_loss'].append(data['eval']['loss'])
        metrics['mse_loss'].append(data['train']['mse_loss'])
        metrics['commitment_loss'].append(data['train']['commitment_loss'])
        metrics['psnr'].append(data['train']['psnr'])
        metrics['ssim'].append(data['train']['ssim'])
        metrics['lpips'].append(data['train']['lpips'])

    # Plotting Training vs Evaluation Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, metrics['train_loss'], label='Training Loss')
    plt.plot(epochs, metrics['eval_loss'], label='Evaluation Loss')
    plt.title('Training vs Evaluation Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(epochs)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(subfolder, 'Training_vs_Eval_Loss.png'), format='png')  # Saving the figure as PNG
    plt.close()

    # Plotting Sub-losses for Training
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, metrics['mse_loss'], label='MSE Loss')
    plt.plot(epochs, metrics['commitment_loss'], label='Commitment Loss')
    plt.plot(epochs, metrics['psnr'], label='PSNR')
    plt.plot(epochs, metrics['ssim'], label='SSIM')
    plt.plot(epochs, metrics['lpips'], label='LPIPS')
    plt.title('Training Sub-losses Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Values')
    plt.xticks(epochs)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(subfolder, 'Training_Sublosses.png'), format='png')  # Saving the figure as PNG
    plt.close()
