import os
from typing import List
import optuna

# @dataclass
# class ModelParams:
#     model_name: str
#     training_type: str  # "diverse" or "prototype"
#     suggested_freeze: int
#     dropout_rate: float
#     lr: float
#     weight_decay: float
#     algorithm: str
#     momentum: float

filepath_root = os.getcwd()
diverse_folder = os.path.join(filepath_root, "studies", "diverse") #for optuna-sql
prototype_folder = os.path.join(filepath_root, "studies", "prototype") #same wie oben

def extract_model_name(filename: str) -> str:
    return os.path.splitext(filename)[0]

class ModelParams():
    "Stores name and place pairs"
    def __init__(self, model_name, training_type, suggested_freeze, dropout_rate, lr, weight_decay, algorithm, momentum, nesterov):
        self.model_name = str(model_name)
        self.training_type = str(training_type)
        self.suggested_freeze = int(suggested_freeze)
        self.dropout_rate = float(dropout_rate)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.algorithm = str(algorithm)
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)

def load_model_params_from_folder(folder=diverse_folder, training_type= "diverse"):
    db_files = [f for f in os.listdir(folder) if f.endswith(".db")]
    model_params_list = []
    for db_file in db_files:
        db_path = os.path.join(folder, db_file)
        storage_url = f"sqlite:///{db_path}"
        study = optuna.load_study(study_name=None, storage=storage_url)

        best = study.best_trial
        params = best.params
        momentum = "momentum"
        match params["algorithm"]:
            case "Adadelta":
                lr = "lr_Adadelta"
                weight_decay = "weight_decay_Adadelta"
            case "Adam":
                lr = "lr_Adam"
                weight_decay = "weight_decay_Adam"
            case "AdamW":
                lr = "lr_AdamW"
                weight_decay = "weight_decay_AdamW"
            case SGD:
                lr = "lr_SGD"
                weight_decay = "weight_decay_SGD"
                momentum = "momentum_SGD"

        extraction = ModelParams(
            model_name=extract_model_name(db_file),
            training_type=training_type,
            suggested_freeze=params["suggested_freeze"],
            dropout_rate=params["dropout_rate"],
            lr=params[lr],
            weight_decay=params.get(weight_decay, 0),
            algorithm=params["algorithm"],
            momentum=params.get(momentum, 0),
            nesterov=params.get("nesterov_SGD", False)
        )
        model_params_list.append(extraction)

    return model_params_list

print(load_model_params_from_folder(diverse_folder, "diverse"))


        # # Dynamically find any parameter that starts with 'lr_'
        # lr_keys = [key for key in params if key.startswith("lr_")]
        # if not lr_keys:
        #     raise ValueError("No parameter found starting with 'lr_'")
        # lr_key = lr_keys[0]  # Use the first matching key
        # lr_value = float(params[lr_key])
        
        # weight_keys = [key for key in params if key.startswith("weight_")]
        # if not lr_keys:
        #     raise ValueError("No parameter found starting with 'lr_'")
        # lr_key = weight_keys[0]  # Use the first matching key
        # weight_value = float(params[lr_key])

        # model_param = ModelParams(
        #     model_name=extract_model_name(db_file),
        #     training_type=training_type,
        #     suggested_freeze=int(params["suggested_freeze"]),


    
