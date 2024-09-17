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
