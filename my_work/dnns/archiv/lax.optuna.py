#Code with optuna implementation. No CSV will be created. Only output
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch import nn, optim

from transformers import CLIPModel, ViTImageProcessor, ViTForImageClassification
import os

import optuna

import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import helper as h

SEED = 42
torch.manual_seed(SEED)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()])

# Define a custom image folder class to return correct indices (needed for logging predictions)
class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        # Call the original ImageFolder __getitem__8
        original_tuple = super(CustomImageFolder, self).__getitem__(index)
        # Add the index to the original tuple
        return (*original_tuple, index)

# Check if CUDA is available and use it; 
# otherwise, check if MPS (for mac) is available; otherwise, use CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


# Get paths for training and testing data
filepath_root = os.getcwd() # looks like: C:\Masterthesis_unibe\my_work\dnns\
n_epochs = 6

test_path = os.path.join(filepath_root, 'dataset', 'test_trials_learning') 

# DNN function main
def train_model (criterion, epoch, model, model_name, optimizer, training):
    
    #set trainings modus: prototype or diverse
    train_path = os.path.join(filepath_root, 'dataset', training)

    # Load training data
    train_data = CustomImageFolder(root=train_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)

    # train for n_epochs
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

    # Calculate training accuracy and loss
    train_accuracy = 100 * correct_train / total_train
    train_loss = running_loss/len(train_loader)
    # Print progress
    print(f"{model_name}, Epoch {epoch+1}, Training Accuracy: {train_accuracy}%, Training Loss: {train_loss}")

def test_model(criterion, epoch, model, model_name):

# Testing phase
    testset_path = test_path + f'/testset_{epoch+1}' 
    test_data = datasets.ImageFolder(root=testset_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False)

    # Set model to evaluation mode
    model.eval()

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

    # Calculate test accuracy and loss
    test_accuracy = 100 * correct_test / total_test
    test_loss = running_test_loss / len(test_loader)
    print(f'{model_name}, Epoch {epoch+1}, Test Accuracy: {test_accuracy}%, Test Loss: {test_loss}')

    return test_loss

training = 'prototype'
model_name= 'alexnet'
  
def objective(trial, training, model_name):

    # Load a pre-trained model and modify the final layer
    model = h.load_model(model_name, device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    learning= trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    algorithm = trial.suggest_categorical("algorithm", ["Adadelta", "Adam"])
    if algorithm == "Adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=learning)
    else: 
        optimizer = optim.Adam(model.parameters(), lr= learning)

    #optimizing loop
    for epoch in range(n_epochs):
        train_model(criterion, epoch, model, model_name, optimizer, training)
        
        loss = test_model(criterion, epoch, model, model_name)
        
        trial.report(loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    return loss


if __name__ == '__main__':
    trainings = ['diverse', 'prototype']
    for training in trainings:
        architectures = ['vgg16', 'alexnet', 'convnext', 'efficientnet']
        for model_name in architectures:
            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED), pruner=optuna.pruners.MedianPruner(),)
            study.optimize(lambda trial: objective(trial, training, model_name), n_trials=10, n_jobs=2)  # Parallel execution

            print(f"Best hyperparameters for {training} with {model_name}: {study.best_params}")



