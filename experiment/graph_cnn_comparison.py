import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import re

# Font settings for matplotlib
plt.rcParams.update({'font.size': 14, 'font.family': 'Consolas'})

def add_fading_effect(ax, x, y, color):
    layers = 4
    alpha_step = 0.1
    offset_step = 0.05
    
    for i in range(1, layers + 1):
        # Calculate lower bound for each layer
        lower_bound = [value - i * offset_step for value in y]
        # Calculate alpha for each layer
        current_alpha = alpha_step * (layers - i + 1)
        # Add fill between
        ax.fill_between(x, lower_bound, y, color=color, alpha=current_alpha)

# Function to load JSON data from a file
def load_json_data(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

# Extract epoch number from filename
def extract_epoch(filename):
    match = re.search(r'(\d+)\.json$', filename)
    if match:
        return int(match.group(1))
    return 0  # Default to 0 if no number is found

# Base directory paths for both methods
base_dirs = {
    'conventional': './traditional/results',
    'hyperdimensional': './hyperdimensional/results'
}

# Collect all subfolders from both directories and find matches
conventional_subfolders = {f for f in os.listdir(base_dirs['conventional']) if os.path.isdir(os.path.join(base_dirs['conventional'], f))}
hyperdimensional_subfolders = {f for f in os.listdir(base_dirs['hyperdimensional']) if os.path.isdir(os.path.join(base_dirs['hyperdimensional'], f))}

# Find matching subfolders
matching_subfolders = conventional_subfolders.intersection(hyperdimensional_subfolders)
matching_subfolders = sorted([msf for msf in matching_subfolders if ("hdc" in msf or "cnn" in msf or "mlp" in msf) and ("stl" in msf)])

# Setup figure and axes
num_subfolders = len(matching_subfolders)
fig, axes = plt.subplots(2, num_subfolders * 3, figsize=(15 * num_subfolders, 8), squeeze=False)

# Process each matching subfolder
for idx, subfolder in enumerate(matching_subfolders):
    conventional_train_acc = []
    hyperdimensional_train_acc = []
    conventional_eval_acc = []
    hyperdimensional_eval_acc = []
    epochs = []

    # Load data from both conventional and hyperdimensional results
    for method, base_dir in base_dirs.items():
        json_files = sorted(glob.glob(os.path.join(base_dir, subfolder, '*.json')), key=extract_epoch)
        for json_file in json_files:
            epoch = extract_epoch(json_file)
            if method == 'conventional' and epoch not in epochs:
                epochs.append(epoch)
            data = load_json_data(json_file)
            if method == 'conventional':
                conventional_train_acc.append(data.get('training_accuracy', 0))
                conventional_eval_acc.append(data.get('eval_accuracy', 0))
            else:
                hyperdimensional_train_acc.append(data.get('training_accuracy', 0))
                hyperdimensional_eval_acc.append(data.get('eval_accuracy', 0))

    # Plot training accuracy
    ax_train = axes[0, idx*3]
    ax_train.plot(epochs, conventional_train_acc, 'r--', label='Conventional Training Accuracy')
    ax_train.plot(epochs, hyperdimensional_train_acc, 'g-', label='Hyperdimensional Training Accuracy')
    ax_train.fill_between(epochs, [acc - 0.05 for acc in conventional_train_acc], conventional_train_acc, color='red', alpha=0.2)
    ax_train.fill_between(epochs, [acc - 0.05 for acc in hyperdimensional_train_acc], hyperdimensional_train_acc, color='green', alpha=0.2)
    ax_train.set_title(f'Train: {subfolder}')
    ax_train.set_xticks(np.arange(1, 11))
    ax_train.set_yticks(np.arange(0.1, 1.1, 0.1))
    ax_train.set_xlabel('Epoch')
    ax_train.set_ylabel('Training Accuracy')

    # Plot evaluation accuracy
    ax_eval = axes[1, idx*3]
    ax_eval.plot(epochs, conventional_eval_acc, 'r--', label='Conventional Eval Accuracy')
    ax_eval.plot(epochs, hyperdimensional_eval_acc, 'g-', label='Hyperdimensional Eval Accuracy')
    ax_eval.fill_between(epochs, [acc - 0.05 for acc in conventional_eval_acc], conventional_eval_acc, color='red', alpha=0.2)
    ax_eval.fill_between(epochs, [acc - 0.05 for acc in hyperdimensional_eval_acc], hyperdimensional_eval_acc, color='green', alpha=0.2)
    ax_eval.set_title(f'Eval: {subfolder}')
    ax_eval.set_xticks(np.arange(1, 11))
    ax_eval.set_yticks(np.arange(0.1, 1.1, 0.1))
    ax_eval.set_xlabel('Epoch')
    ax_eval.set_ylabel('Eval Accuracy')

    # Load confusion matrices and plot them
    short_hand = ["Conv.", "HD"]
    for i, method in enumerate(['conventional', 'hyperdimensional']):
        cm_path_train = os.path.join(base_dirs[method], subfolder, f'confusion_matrix_train_epoch_10.csv')
        cm_path_eval = os.path.join(base_dirs[method], subfolder, f'confusion_matrix_eval_epoch_10.csv')
        
        cm_train = pd.read_csv(cm_path_train, index_col=0)
        cm_train = cm_train.div(cm_train.sum(axis=1), axis=0)
        ax_cm_train = axes[0, idx*3 + i + 1]
        im = ax_cm_train.imshow(cm_train, cmap='Blues', aspect='equal', interpolation='nearest')
        ax_cm_train.set_title(f'{short_hand[i]} Train Heatmap')
        ax_cm_train.set_xticks(np.arange(10))
        ax_cm_train.set_yticks(np.arange(10))
        ax_cm_train.set_xlabel('Predicted')
        ax_cm_train.set_ylabel('Actual')
        ax_cm_train.set_aspect('equal', adjustable='box')
        for (j, k), val in np.ndenumerate(cm_train.values):
            ax_cm_train.text(k, j, f"{val:.2f}", ha='center', va='center', color='white' if im.norm(val) > 0.5 else 'black', fontsize=7)

        cm_eval = pd.read_csv(cm_path_eval, index_col=0)
        cm_eval = cm_eval.div(cm_eval.sum(axis=1), axis=0)
        ax_cm_eval = axes[1, idx*3 + i + 1]
        im = ax_cm_eval.imshow(cm_eval, cmap='Blues', aspect='equal', interpolation='nearest')
        ax_cm_eval.set_title(f'{short_hand[i]} Eval Heatmap')
        ax_cm_eval.set_xticks(np.arange(10))
        ax_cm_eval.set_yticks(np.arange(10))
        ax_cm_eval.set_xlabel('Predicted')
        ax_cm_eval.set_ylabel('Actual')
        ax_cm_eval.set_aspect('equal', adjustable='box')
        for (j, k), val in np.ndenumerate(cm_eval.values):
            ax_cm_eval.text(k, j, f"{val:.2f}", ha='center', va='center', color='white' if im.norm(val) > 0.5 else 'black', fontsize=7)


# Add legend to the plots
handles, labels = ax_train.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower right', fontsize=12)

# Adjust subplots
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15, wspace=0.35, hspace=0.35)
plt.savefig('comparison_with_heatmaps.png', format='png', dpi=1200)
plt.close()
