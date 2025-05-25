import helper as h


import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


#'alexnet', 'convnext', 'efficientnet', 'resnet'
architectures = ['vgg']
for model_name in architectures:
    dropout_rate = 0
    model = h.load_model(model_name, device, dropout_rate)
    h.freezer(model, 0)
    for name, layer in model.named_modules():
        print(name, ":", layer.__class__.__name__)