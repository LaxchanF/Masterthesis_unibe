
import os
from PIL import Image as PILImage
import torchvision.models as models
from datasets import Dataset, DatasetDict, Features, ClassLabel, Image, Value
from transformers import ViTImageProcessor
from torch import nn

######################################
#######General Helper Functions#######
######################################

def freezer(model, suggested_freeze):
    # Extract layers and exclude the final classifier layer (so we don't freeze it)
    if isinstance(model, models.ResNet):
        #layers = list(model.children())[:-1] 
        layers = list(model.children())
         # Exclude the classifier (fc layer)
    elif isinstance(model, models.VGG):
        layers = list(model.features)  # Use the feature layers of VGG
    elif isinstance(model, models.EfficientNet):
        layers = list(model.features)  # EfficientNet feature layers
    elif isinstance(model, models.AlexNet):
        layers = list(model.features)  # AlexNet feature layers
    elif isinstance(model, models.ConvNeXt):
        layers = list(model.features)  # ConvNeXt feature layers
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


    # Limit freezing to the number of layers available or the suggested number
    num_layers_to_freeze = min(len(layers), suggested_freeze)
    
    # Freeze the layers
    layer_count = 0
    for idx, layer in enumerate(layers):
        if layer_count < num_layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
            # print(f"Froze layer {idx}: {layer.__class__.__name__}")
            layer_count += 1
        else:
            # print(f"Left layer {idx} trainable: {layer.__class__.__name__}")
            pass
    return model

#inshallah code
def dir_up(path, levels):
    for _ in range(levels):
        path = os.path.dirname(path)
    return path

# Handy function for Mac users to get rid of .DS_Store files
def get_rid_of_DS_Store(path):
    for file_name in os.listdir(path):
        if file_name.split('/')[-1] == ".DS_Store":
            os.remove(path + file_name)

######################################
#######Model loading for main.py#######
######################################

# This function loads the model based on the model name + LF: implemented Drop_out rate as variable for optuna
def load_model(model_name, device, dropout_rate):
        if model_name.startswith('resnet'):
            model = models.resnet50(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(dropout_rate),  # Add dropout with the specified rate
                nn.Linear(num_ftrs, 3)
                )
        elif model_name.startswith('vgg'):
            model = models.vgg16(pretrained=True)
            num_ftrs = model.classifier[6].in_features 
            model.classifier[6] = nn.Sequential(
                nn.Dropout(dropout_rate),  # Add dropout with the specified rate
                nn.Linear(num_ftrs, 3)
                )
        elif model_name.startswith('eff'):
            model = models.efficientnet_v2_m(pretrained=True)
            num_ftrs = model.classifier[1].in_features  
            model.classifier[1] = nn.Sequential(
                nn.Dropout(dropout_rate),  # Add dropout with the specified rate
                nn.Linear(num_ftrs, 3)
                )
        elif model_name.startswith('alex'):
            model = models.alexnet(pretrained=True)
            num_ftrs = model.classifier[6].in_features 
            model.classifier[6] = nn.Sequential(
                nn.Dropout(dropout_rate),  # Add dropout with the specified rate
                nn.Linear(num_ftrs, 3)
                )
        elif model_name.startswith('conv'):
            model = models.convnext_base(pretrained=True)
            num_ftrs = model.classifier[2].in_features  
            model.classifier[2] = nn.Sequential(
                nn.Dropout(dropout_rate),  # Add dropout with the specified rate
                nn.Linear(num_ftrs, 3)
                )
        model = model.to(device)
        return model

######################################
#######Data loading for vit.py#######
######################################

def load_data():
    # Get paths for training and testing data
    cwd = os.getcwd()
    train_path = os.path.join(cwd, 'dataset', 'train')
    test_path_1 = os.path.join(cwd, 'dataset', 'test', 'testset_1')
    test_path_2 = os.path.join(cwd, 'dataset', 'test', 'testset_2')
    test_path_3 = os.path.join(cwd, 'dataset', 'test', 'testset_3')
    test_path_4 = os.path.join(cwd, 'dataset', 'test', 'testset_4')
    test_path_5 = os.path.join(cwd, 'dataset', 'test', 'testset_5')
    test_path_6 = os.path.join(cwd, 'dataset', 'test', 'testset_6')

    # Load images with filenames
    def load_images_with_filenames(data_dir):
        get_rid_of_DS_Store(data_dir + '/')
        images, labels, filenames = [], [], []
        label_set = set()  # To store unique labels
        for label in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, label)
            get_rid_of_DS_Store(class_dir+ '/')
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    file_path = os.path.join(class_dir, filename)
                    image = PILImage.open(file_path)
                    images.append(image)
                    labels.append(label)
                    filenames.append(filename)
                    label_set.add(label)
        return images, labels, filenames, sorted(label_set)

    def preprocess(example):
        # Get the image directly
        image = example['image']

        # Resize the image
        resized_image = image.resize((224, 224))

        # Convert image to RGB if it's in RGBA mode
        if resized_image.mode == 'RGBA':
            resized_image = resized_image.convert('RGB')

        # Update the example
        example['image'] = resized_image

        return example


    model_name_or_path = 'google/vit-base-patch16-224-in21k'
    processor = ViTImageProcessor.from_pretrained(model_name_or_path)
    def transform(example_batch):
        # Take a list of PIL images and turn them to pixel values
        inputs = processor([x for x in example_batch['image']], return_tensors='pt')

        # Don't forget to include the labels!
        inputs['label'] = example_batch['label']
        inputs['filename'] = example_batch['filename'] 
        return inputs

    # Load images, labels, and filenames for train and test sets
    train_images, train_labels, train_filenames, label_names = load_images_with_filenames(train_path)
    # test_images, test_labels, test_filenames, _ = load_images_with_filenames(test_path_1)
    test_images_1, test_labels_1, test_filenames_1, _ = load_images_with_filenames(test_path_1)
    test_images_2, test_labels_2, test_filenames_2, _ = load_images_with_filenames(test_path_2)
    test_images_3, test_labels_3, test_filenames_3, _ = load_images_with_filenames(test_path_3)
    test_images_4, test_labels_4, test_filenames_4, _ = load_images_with_filenames(test_path_4)
    test_images_5, test_labels_5, test_filenames_5, _ = load_images_with_filenames(test_path_5)
    test_images_6, test_labels_6, test_filenames_6, _ = load_images_with_filenames(test_path_6)

    # Define the features of the dataset
    features = Features({
        'image': Image(decode=True),
        'label': ClassLabel(names=label_names),
        'filename': Value('string')
    })

    # Create the datasets with the defined features
    train_dataset = Dataset.from_dict({'image': train_images, 'label': train_labels, 'filename': train_filenames}, features=features)
    # test_dataset = Dataset.from_dict({'image': test_images, 'label': test_labels, 'filename': test_filenames}, features=features)
    test_dataset_1 = Dataset.from_dict({'image': test_images_1, 'label': test_labels_1, 'filename': test_filenames_1}, features=features)
    test_dataset_2 = Dataset.from_dict({'image': test_images_2, 'label': test_labels_2, 'filename': test_filenames_2}, features=features)
    test_dataset_3 = Dataset.from_dict({'image': test_images_3, 'label': test_labels_3, 'filename': test_filenames_3}, features=features)
    test_dataset_4 = Dataset.from_dict({'image': test_images_4, 'label': test_labels_4, 'filename': test_filenames_4}, features=features)
    test_dataset_5 = Dataset.from_dict({'image': test_images_5, 'label': test_labels_5, 'filename': test_filenames_5}, features=features)
    test_dataset_6 = Dataset.from_dict({'image': test_images_6, 'label': test_labels_6, 'filename': test_filenames_6}, features=features)

    # # Combine into a DatasetDict
    # dataset = DatasetDict({
    #     'train': train_dataset,
    #     'test': test_dataset
    # })

    # Combine into a DatasetDict
    dataset = DatasetDict({
        'train': train_dataset,
        'test_1': test_dataset_1,
        'test_2': test_dataset_2,
        'test_3': test_dataset_3,
        'test_4': test_dataset_4,
        'test_5': test_dataset_5,
        'test_6': test_dataset_6
    })
    # Preprocess the images
    dataset = dataset.map(preprocess)
    prepared_dataset = dataset.with_transform(transform)

    return prepared_dataset