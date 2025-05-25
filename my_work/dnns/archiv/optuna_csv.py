import torchvision.models as models
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch import nn, optim
import optuna
from optuna.exceptions import TrialPruned
import csv  # Importing the CSV module

import helper as h

# --- Create Folders for Saving ---
os.makedirs("models", exist_ok=True)
os.makedirs("studies", exist_ok=True)

# --- Global Settings ---
SEED = 42
torch.manual_seed(SEED)

# --- Device Setup ---
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# --- Constants ---
filepath_root = os.getcwd()
testset_root = os.path.join(filepath_root, 'dataset', 'test_trials_learning')
n_epochs = 6
batch_size = 4

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --- Custom Dataset Class ---
class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        return (*original_tuple, index)

# --- Training Function ---
def train_model(criterion, epoch, model, model_name, optimizer, train_loader, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels, _ in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        if torch.isnan(loss):
            raise ValueError("NaN loss encountered. Pruning trial...") 

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    avg_loss = running_loss / len(train_loader)
    print(f"{model_name}, Epoch {epoch+1}, Train Acc: {acc:.2f}%, Train Loss: {avg_loss:.4f}")
    return acc, avg_loss

# --- Testing Function ---
def test_model(criterion, epoch, model, model_name, test_loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    avg_loss = running_loss / len(test_loader)
    print(f"{model_name}, Epoch {epoch+1}, Test Acc: {acc:.2f}%, Test Loss: {avg_loss:.4f}")
    return acc, avg_loss

def objective(trial, device, model_name, train_loader, test_loaders, study_csv_writer):

    # Freezing layers (only feature layers)
    if model_name.startswith('resnet'):
        suggested_freeze = trial.suggest_int("suggested_freeze", 0, 31)
    elif model_name.startswith('vgg'):
        suggested_freeze = trial.suggest_int("suggested_freeze", 0, 31)
    elif model_name.startswith('eff'):
        suggested_freeze = trial.suggest_int("suggested_freeze", 0, 9)
    elif model_name.startswith('alex'):
        suggested_freeze = trial.suggest_int("suggested_freeze", 0, 13)
    else:
        suggested_freeze = trial.suggest_int("suggested_freeze", 0, 8)

    dropout_rate = trial.suggest_float("dropout_rate", 0, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    # Optimizer setup
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    algorithm = trial.suggest_categorical("algorithm", ["Adadelta", "Adam", "AdamW", "SGD"])

    model = h.load_model(model_name, device, dropout_rate)
    h.freezer(model, suggested_freeze)

    criterion = nn.CrossEntropyLoss()

    if algorithm == "Adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
    elif algorithm == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif algorithm == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif algorithm == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    best_accuracy = 0.0
    best_model_state = None

    for epoch in range(n_epochs):
        train_acc, train_loss = train_model(criterion, epoch, model, model_name, optimizer, train_loader, device)
        test_loader = test_loaders[epoch]
        test_acc, test_loss = test_model(criterion, epoch, model, model_name, test_loader, device)

        if torch.isnan(torch.tensor(test_loss)):
            raise TrialPruned()

        trial.report(test_acc, epoch)

        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model_state = model.state_dict()

        if trial.should_prune():
            raise TrialPruned()

    # Save the best model at the end of the trial
    model_save_path = os.path.join(filepath_root, "models", f"best_model_{model_name}_trial_{trial.number}.pth")
    torch.save(best_model_state, model_save_path)

    # Save the trial's results to CSV
    trial_results = {
        'trial_number': trial.number,
        'value': best_accuracy,
        'lr': lr,
        'dropout_rate': dropout_rate,
        'weight_decay': weight_decay,
        'algorithm': algorithm,
        'suggested_freeze': suggested_freeze
    }

    study_csv_writer.writerow(trial_results)  # Writing the row to the CSV

    return best_accuracy

if __name__ == '__main__':
    # Create CSV for storing the results
    csv_file_path = os.path.join("studies", "study_results.csv")
    with open(csv_file_path, mode='w', newline='') as file:
        csv_writer = csv.DictWriter(file, fieldnames=["trial_number", "value", "lr", "dropout_rate", "weight_decay", "algorithm", "suggested_freeze"])
        csv_writer.writeheader()  # Write the header

        trainings = ['diverse', 'prototype']
        architectures = ['vgg16', 'alexnet', 'convnext', 'efficientnet', 'resnet']

        # Create train_loaders and test_loaders...
        # Your code for creating data loaders goes here...

        for training in trainings:
            for model_name in architectures:
                print(f"\n🚀 Optimizing {model_name} on {training} training set")

                # ✅ Optimize with cached model & loaders
                study = optuna.create_study(
                    direction="maximize",
                    sampler=optuna.samplers.TPESampler(seed=SEED),
                    pruner=optuna.pruners.MedianPruner()
                )

                study.optimize(
                    lambda trial: objective(trial, device, model_name, train_loaders[training], test_loaders, csv_writer),
                    n_trials=10,
                    n_jobs=1  # Theoretically parallel job execution. Too complex right now..
                )

                print(f"✅ Best hyperparameters for {model_name} on {training}: {study.best_params}")
