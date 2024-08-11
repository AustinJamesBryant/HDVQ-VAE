import os
import json
import matplotlib.pyplot as plt
import glob

# Font settings for matplotlib
plt.rcParams.update({'font.size': 18, 'font.family': 'Consolas'})

# Function to load JSON data from a file
def load_json_data(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

# Path to the results directory for the classifier
results_dir = './results/hdc_stl_classifier'

# Collect all JSON files in the results directory
json_files = glob.glob(os.path.join(results_dir, '*.json'))

# Data dictionaries
epoch_data = {}

# Load each JSON file and extract the necessary data
for json_file in json_files:
    data = load_json_data(json_file)
    epoch_number = int(json_file.split('_')[-1].split('.')[0])  # Extract epoch number from filename
    epoch_data[epoch_number] = data

# Sorted epochs and their corresponding accuracies
epochs = sorted(epoch_data.keys())
training_accuracies = [epoch_data[epoch]['training_accuracy'] for epoch in epochs]
eval_accuracies = [epoch_data[epoch]['eval_accuracy'] for epoch in epochs]

# Plotting Training and Evaluation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(epochs, training_accuracies, 'g-', label='Training Accuracy')
plt.plot(epochs, eval_accuracies, 'r--', label='Evaluation Accuracy')
plt.fill_between(epochs, [acc - 0.05 for acc in eval_accuracies],eval_accuracies, color='red', alpha=0.2)
plt.fill_between(epochs, [acc - 0.05 for acc in training_accuracies], training_accuracies, color='green', alpha=0.2)
plt.title('HDC Classification Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(range(1, 11))  # Setting x-ticks to show epochs 1-10
plt.yticks([i/10 for i in range(11)])  # Setting y-ticks to range from 0 to 1
plt.ylim(0, 1)  # Limiting y-axis from 0 to 1
plt.legend(loc='lower left')
plt.grid(False)
plt.savefig(os.path.join(results_dir, 'Accuracy_over_epochs.png'), format='png', dpi=1200)  # Saving the figure as PNG
plt.show()
