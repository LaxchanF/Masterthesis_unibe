import torchvision.models as models
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch import nn, optim
import optuna
from optuna.exceptions import TrialPruned

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

        # Optimizing. Auto cast if float16 or float 32 usw. Funktioniert nicht wie gewollt... rausnehmen?
        #with autocast(device_type="cuda"):
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        if torch.isnan(loss):
            raise TrialPruned("NaN loss encountered. Pruning trial...") 

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    avg_loss = running_loss / len(train_loader)
    #print(f"{model_name}, Epoch {epoch+1}, Train Acc: {acc:.2f}%, Train Loss: {avg_loss:.4f}")
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
    #print(f"{model_name}, Epoch {epoch+1}, Test Acc: {acc:.2f}%, Test Loss: {avg_loss:.4f}")
    return acc, avg_loss


def objective(trial, device, model_name, train_loader, test_loaders):

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

    #dropout rate passed to helper function load_model()
    dropout_rate = trial.suggest_float("dropout_rate", 0, 0.5)



    #Optimizer stuff
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    momentum = trial.suggest_float("momentum", 0.5, 0.99)



    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    model = h.load_model(model_name, device, dropout_rate)
    h.freezer(model, suggested_freeze)

    criterion = nn.CrossEntropyLoss()

    #optimizer = optim.Adam(model.parameters(), lr=lr)
    algorithm = trial.suggest_categorical("algorithm", ["Adam", "AdamW", "SGD", "SGD_Nesterov"])

    if algorithm == "Adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
    elif algorithm == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr) #weight_decay=weight_decay rausnehmen. adam sensitiv af
    elif algorithm == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif algorithm == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif algorithm == "SGD_Nesterov":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)


    for epoch in range(n_epochs):
        train_acc, train_loss = train_model(criterion, epoch, model, model_name, optimizer, train_loader, device)
        test_loader = test_loaders[epoch]
        test_acc, test_loss = test_model(criterion, epoch, model, model_name, test_loader, device)


        if torch.isnan(torch.tensor(test_loss)):
            raise TrialPruned("NaN loss encountered. Pruning trial...") 
        
        trial.report(test_acc, epoch)    

        if trial.should_prune():
            raise TrialPruned()

    return test_acc


if __name__ == '__main__':
    trainings = ['diverse', 'prototype']
    architectures = ['vgg16', 'alexnet', 'convnext', 'efficientnet', 'resnet']

    # Create a dict with Prototype and Diverse. Key = name (div. or proto.) Pass only right Data into Objective function
    train_loaders = {}
    for training in trainings:
        train_path = os.path.join(filepath_root, 'dataset', training)
        train_data = CustomImageFolder(root=train_path, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        train_loaders[training] = train_loader

    # Create a List with Testingdata. Pass whole list to objective 
    test_loaders = []
    for epoch in range(1, n_epochs + 1):
        testset_path = os.path.join(testset_root, f'testset_{epoch}')
        test_data = datasets.ImageFolder(root=testset_path, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
        test_loaders.append(test_loader)

    for training in trainings:
        for model_name in architectures:
            print(f"\n🚀 Optimizing {model_name} on {training} training set")

            study_name = f"{model_name}_{training}_study"
            storage_path = os.path.join(filepath_root, "studies", f"{study_name}.db")
            storage_url = f"sqlite:///{storage_path}"

            study = optuna.create_study(
                study_name=study_name,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=SEED),
                pruner=optuna.pruners.MedianPruner(),
                storage=storage_url,
                load_if_exists=True  # So it can resume if already exists
            )

            # # ✅ Optimize with cached model & loaders
            # study = optuna.create_study(
            #     direction="maximize",
            #     sampler=optuna.samplers.TPESampler(seed=SEED),
            #     pruner=optuna.pruners.MedianPruner()
            # )

            study.optimize(
                lambda trial: objective(trial, device, model_name, train_loaders[training], test_loaders),
                n_trials=1000,
                n_jobs=1 # Theoretisch paralell jobs execution. Too complex rn..
            )

            # print(f"🏁 Best trial test accuracy: {study.best_trial.value:.2f}")
            # print(f"🔧 Best params: {study.best_trial.params}")
