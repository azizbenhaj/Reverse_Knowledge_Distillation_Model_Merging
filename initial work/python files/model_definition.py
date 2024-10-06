import torch
from torchvision import models
from torch import nn


# Code for loading and modifying the Vision Transformer (ViT) model

num_classes = 10

def get_vit_b_16_model(num_classes=num_classes):
    # Load the pre-trained Vision Transformer (ViT) model
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)  # Using pre-trained weights

    # Modify the classifier to fit MNIST's 10 classes instead of 1000 ImageNet classes
    in_features = model.heads[-1].in_features
    model.heads = nn.Linear(in_features=in_features, out_features=num_classes)

    # Move the model to GPU if available
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps")
    model.to(device)

    return model, device

def get_vit_b_32_model(num_classes=num_classes):
    # Load the pre-trained Vision Transformer (ViT) model
    model = models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1)  # Using pre-trained weights

    # Modify the classifier to fit MNIST's 10 classes instead of 1000 ImageNet classes
    in_features = model.heads[-1].in_features
    model.heads = nn.Linear(in_features=in_features, out_features=num_classes)

    # Move the model to GPU if available
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps")
    model.to(device)

    return model, device

# Save the model's state dictionary
def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_pretrained_model(base_model, path, output_classes=10, device=torch.device("mps")):
    # Initialize the teacher model
    teacher_model = base_model
    #device = torch.device("cuda")
    device = device
    teacher_model.to(device)

    # Modify the classifier to fit MNIST's 10 classes
    last_layer = teacher_model.heads[-1]
    in_features = last_layer.in_features
    teacher_model.heads = nn.Linear(in_features=in_features, out_features=output_classes)

    # Load the trained model
    teacher_model.load_state_dict(torch.load(path, map_location=device))
    return teacher_model