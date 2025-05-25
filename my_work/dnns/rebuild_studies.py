import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch import nn, optim
from dataclasses import dataclass
from typing import List
from transformers import CLIPModel, ViTImageProcessor, ViTForImageClassification
import csv
import os
import help_rebuild as h

# --- Constants ---
filepath_root = os.getcwd()
diverse_folder = os.path.join(filepath_root, "studies", "diverse") #for optuna-sql
prototype_folder = os.path.join(filepath_root, "studies", "prototype") #same wie oben
testset_root = os.path.join(filepath_root, 'dataset', 'test_trials_learning')
batch_size = 4

diverse_params: List[h.ModelParams] = h.load_model_params_from_folder(diverse_folder, "diverse")
prototype_params: List[h.ModelParams] = h.load_model_params_from_folder(prototype_folder, "prototype")


os.makedirs('Auswertung', exist_ok=True)
SEED = 42
torch.manual_seed(SEED)

# Define transformations with data augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()])


# Create a List with Testingdata. Pass whole list to objective 
test_loaders = []
for epoch in range(1, 6 + 1):
    testset_path = os.path.join(testset_root, f'testset_{epoch}')
    test_data = datasets.ImageFolder(root=testset_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    test_loaders.append(test_loader)

# Check if CUDA is available and use it; 
# otherwise, check if MPS (for mac) is available; otherwise, use CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Define a custom image folder class to return correct indices (needed for logging predictions)
class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        # Call the original ImageFolder __getitem__
        original_tuple = super(CustomImageFolder, self).__getitem__(index)
        # Add the index to the original tuple
        return (*original_tuple, index)

# Define loss function
criterion = nn.CrossEntropyLoss()


from dataclasses import dataclass
from typing import List

# create params from best optuna trial
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


def get_params_from_lists(model_name: str, training_type: str) -> h.ModelParams:
    params_list = diverse_params if training_type == "diverse" else prototype_params
    for param in params_list:
        if model_name in param.model_name:
            return param
    raise ValueError(f"Parameters for model '{model_name}' and training type '{training_type}' not found.")


# DNN function main
def main (model_name='vgg16',
          jez_trainiert='prototype',
          n_epochs=6, n_runs=20,
          save_checkpoints=True,
          train_data = None,
          params = None
          ):

    # Get path for saving checkpoints if save_checkpoints is True
    if save_checkpoints:
        checkpoint_folder = os.path.join(os.getcwd(), 'checkpoints', f'{model_name}_{jez_trainiert}_checkpoints')
        os.makedirs(checkpoint_folder, exist_ok=True)

    # Create a new CSV file for logging predictions ##Lax: '/data/' anpassen zu '/Auswertung/'+ f'{model_name}_diverse_2.csv'
    file_name = os.path.join(filepath_root, 'Auswertung', f'{model_name}_{jez_trainiert}.csv')
    prediction_file = open(file_name, 'w', newline='')
    prediction_writer = csv.writer(prediction_file)

    # Write the header of the CSV file
    prediction_writer.writerow(['run', 'epoch', 'phase', 'image_name', 'ground_truth', 'prediction'])

    # Fine-tune the model for n_runs
    for run in range(n_runs):
        # Load a pre-trained model and modify the final layer
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


        # train and test the model for n_epochs
        for epoch in range(n_epochs):
            train_loader = train_loaders[jez_trainiert]
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

                # Log predictions using indices
                for i, index in enumerate(indices):
                    image_name = train_loader.dataset.imgs[index][0]
                    ground_truth = train_data.classes[labels[i].item()]
                    prediction = train_data.classes[predicted[i].item()]
                    prediction_writer.writerow([run+1, epoch+1, 'train', image_name.split('/')[-1], ground_truth, prediction])

            # Calculate training accuracy and loss
            train_accuracy = 100 * correct_train / total_train
            train_loss = running_loss/len(train_loader)
            # Print progress
            print(f"{model_name}, Run {run+1}, Epoch {epoch+1}, Training Accuracy: {train_accuracy}%, Training Loss: {train_loss}")

            # Save checkpoint if True
            if save_checkpoints:
                checkpoint_path = checkpoint_folder + f'/checkpoint_run_{run+1}_epoch{epoch+1}.pth'
                os.makedirs(checkpoint_folder, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)

            # Testing phase
            test_loader = test_loaders[epoch]

            # Set model to evaluation mode
            model.eval()
            with torch.inference_mode(): #optimizing
  
              running_test_loss = 0.0
              correct_test = 0
              total_test = 0
              image_index = 0
              with torch.no_grad():
                  for inputs, labels in test_loader:
                      inputs, labels = inputs.to(device), labels.to(device)
                      outputs = model(inputs)
                      loss = criterion(outputs, labels)
                      running_test_loss += loss.item()
                      _, predicted = torch.max(outputs.data, 1)
                      total_test += labels.size(0)
                      correct_test += (predicted == labels).sum().item()
  
                      # Log predictions
                      for i in range(labels.size(0)):
                          image_name = test_loader.dataset.imgs[image_index][0]
                          ground_truth = test_data.classes[labels[i].item()]
                          prediction = test_data.classes[predicted[i].item()]
                          prediction_writer.writerow([run+1, epoch+1, 'test', image_name.split('/')[-1], ground_truth, prediction])
                          image_index += 1

            # Calculate test accuracy and loss
            test_accuracy = 100 * correct_test / total_test
            test_loss = running_test_loss / len(test_loader)
            # Print progress
            print(f'{model_name}, Run {run+1}, Epoch {epoch+1}, Test Accuracy: {test_accuracy}%, Test Loss: {test_loss}')

    # Close prediction log file
    prediction_file.close()

if __name__ == '__main__':
    trainings = ['diverse', 'prototype']
    train_loaders = {}
    for training in trainings:
        train_path = os.path.join(filepath_root, 'dataset', training)
        train_data = CustomImageFolder(root=train_path, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        train_loaders[training] = train_loader
        architectures = ['vgg16', 'alexnet', 'convnext', 'efficientnet', 'resnet']
        for model in architectures:
            params = get_params_from_lists(model, training)
            main(model_name=model,
                 jez_trainiert=training,
                 n_epochs=6,
                 n_runs=20,
                 save_checkpoints=False,
                 train_data = train_data,
                 params = params)
        #, 'vgg16', 'alexnet', 'convnext', 'efficientnet'



