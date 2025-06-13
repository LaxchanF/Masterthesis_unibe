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


def get_params_from_lists(model_name: str, training_type: str) -> ModelParams:
    params_list = diverse_params if training_type == "diverse" else prototype_params
    for param in params_list:
        if model_name in param.model_name:
            return param
    raise ValueError(f"Parameters for model '{model_name}' and training type '{training_type}' not found.")



trainings = ['diverse', 'prototype']
for training in trainings:
    architectures = ['vgg16', 'alexnet', 'convnext', 'efficientnet', 'resnet']
    for model in architectures:
        params = get_params_from_lists(model, training)
        print(f"{model}{training})
