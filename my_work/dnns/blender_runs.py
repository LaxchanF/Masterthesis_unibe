import os
import csv
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch import nn, optim
from dataclasses import dataclass
from typing import List
from transformers import CLIPModel, ViTImageProcessor, ViTForImageClassification
import help_rebuild as h

# --- Constants ---
filepath_root = os.getcwd()
diverse_folder = os.path.join(filepath_root, "studies", "diverse")
prototype_folder = os.path.join(filepath_root, "studies", "prototype")
batch_size = 4

# Load model parameters
diverse_params: List[h.ModelParams] = h.load_model_params_from_folder(diverse_folder, "diverse")
prototype_params: List[h.ModelParams] = h.load_model_params_from_folder(prototype_folder, "prototype")

# Create output directory
os.makedirs('Blender_Auswertung', exist_ok=True)

# Set random seed
SEED = 42
torch.manual_seed(SEED)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Custom ImageFolder to also return index
class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(CustomImageFolder, self).__getitem__(index)
        return (*original_tuple, index)

# Loss function
criterion = nn.CrossEntropyLoss()

# Model parameters dataclass
@dataclass
class ModelParams:
    model_name: str
    training_type: str  # "diverse" or "prototype"
    suggested_freeze: int
    dropout_rate: float
    lr: float
    weight_decay: float
    algorithm: str
    momentum: float

# Retrieve parameters

def get_params_from_lists(model_name: str, training_type: str) -> h.ModelParams:
    params_list = diverse_params if training_type == "diverse" else prototype_params
    for param in params_list:
        if model_name in param.model_name:
            return param
    raise ValueError(f"Parameters for model '{model_name}' and training type '{training_type}' not found.")

# Main training function
def main(model_name='vgg16',
         jez_trainiert='prototype',
         n_epochs=6,
         n_runs=20,
         save_checkpoints=True,
         train_data=None,
         train_loader = None,
         test_data=None,
         test_loader = None,
         params=None):

    # Prepare checkpoint folder
    if save_checkpoints:
        checkpoint_folder = os.path.join(filepath_root, 'checkpoints', f'{model_name}_{jez_trainiert}_checkpoints')
        os.makedirs(checkpoint_folder, exist_ok=True)

    # Open prediction CSV file
    file_name = os.path.join(filepath_root, 'Blender_Auswertung', f'{model_name}_{jez_trainiert}.csv')
    with open(file_name, 'w', newline='') as prediction_file:
        prediction_writer = csv.writer(prediction_file)
        prediction_writer.writerow(['run', 'epoch', 'phase', 'image_name', 'ground_truth', 'prediction'])

        for run in range(n_runs):
            model = h.load_model(model_name, device, params.dropout_rate)
            h.freezer(model, params.suggested_freeze)

            # Define optimizer
            algorithm = params.algorithm
            if algorithm == "Adadelta":
                optimizer = optim.Adadelta(model.parameters(), lr=params.lr)
            elif algorithm == "Adam":
                optimizer = optim.Adam(model.parameters(), lr=params.lr)
            elif algorithm == "AdamW":
                optimizer = optim.AdamW(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
            elif algorithm == "SGD":
                optimizer = optim.SGD(model.parameters(), lr=params.lr, momentum=params.momentum, weight_decay=params.weight_decay)
            elif algorithm == "SGD_Nesterov":
                optimizer = optim.SGD(model.parameters(), lr=params.lr, momentum=params.momentum, weight_decay=params.weight_decay, nesterov=True)

            # Training loop
            train_loader = train_loaders[jez_trainiert]
            for epoch in range(n_epochs):
                model.train()
                running_loss = 0.0
                correct_train = 0
                total_train = 0

                for inputs, labels, indices in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_train += labels.size(0)
                    correct_train += (predicted == labels).sum().item()

                    # Log training predictions
                    for i, index in enumerate(indices):
                        image_name = train_loader.dataset.imgs[index][0]
                        ground_truth = train_loader.dataset.classes[labels[i].item()]
                        prediction = train_loader.dataset.classes[predicted[i].item()]
                        prediction_writer.writerow([run+1, epoch+1, 'train', os.path.basename(image_name), ground_truth, prediction])

                train_accuracy = 100 * correct_train / total_train
                train_loss = running_loss / len(train_loader)
                print(f"{model_name}, Run {run+1}, Epoch {epoch+1}, Training Accuracy: {train_accuracy:.2f}%, Training Loss: {train_loss:.4f}")

                # Save checkpoint
                if save_checkpoints:
                    checkpoint_path = os.path.join(checkpoint_folder, f'checkpoint_run_{run+1}_epoch{epoch+1}.pth')
                    torch.save(model.state_dict(), checkpoint_path)

                # Testing phase
                test_loader = test_loaders[jez_trainiert]
                model.eval()

                running_test_loss = 0.0
                correct_test = 0
                total_test = 0

                with torch.inference_mode():
                    for inputs, labels, indices in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        running_test_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total_test += labels.size(0)
                        correct_test += (predicted == labels).sum().item()

                        # Log test predictions
                        for i, index in enumerate(indices):
                            image_name = test_loader.dataset.imgs[index][0]
                            ground_truth = test_loader.dataset.classes[labels[i].item()]
                            prediction = test_loader.dataset.classes[predicted[i].item()]
                            prediction_writer.writerow([run+1, epoch+1, 'test', os.path.basename(image_name), ground_truth, prediction])

                test_accuracy = 100 * correct_test / total_test
                test_loss = running_test_loss / len(test_loader)
                print(f"{model_name}, Run {run+1}, Epoch {epoch+1}, Test Accuracy: {test_accuracy:.2f}%, Test Loss: {test_loss:.4f}")

# Run training/testing
if __name__ == '__main__':
    trainings = ['diverse', 'prototype']
    architectures = ['vgg16', 'alexnet', 'convnext', 'efficientnet', 'resnet']

    train_loaders = {}
    test_loaders = {}

    for training in trainings:
        # Training Data
        train_path = os.path.join(filepath_root, 'blender_dataset', 'Training', training)
        train_data = CustomImageFolder(root=train_path, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        train_loaders[training] = train_loader

        # Testing Data
        test_path = os.path.join(filepath_root, 'blender_dataset', 'Testing', training)
        test_data = CustomImageFolder(root=test_path, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
        test_loaders[training] = test_loader

        for model in architectures:
            params = get_params_from_lists(model, training)
            main(model_name=model,
                 jez_trainiert=training,
                 n_epochs=6,
                 n_runs=1,
                 save_checkpoints=False,
                 train_data=train_data,
                 train_loader= train_loaders[training],
                 test_data=test_data,
                 test_loader = test_loaders[training],
                 params=params)