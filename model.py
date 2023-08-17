'''
models:
Resnet 18, 34, 50, 101
DenseNet
EfficientNetV2
MobileNetV3
'''


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms


def getModel(config):

    if config['model'] == 'resnet50':
        # Load the pre-trained ResNet-50 model
        model = models.resnet50(pretrained=True)

    if config['model'] == 'resnet34':
        # Load the pre-trained ResNet-34 model
        model = models.resnet34(pretrained=True)

    
    if config['model'] == 'resnet18':
        # Load the pre-trained ResNet-18 model
        model = models.resnet18(pretrained=True)

    
    if config['model'] == 'resnet101':
        # Load the pre-trained ResNet-101 model
        model = models.resnet101(pretrained=True)

    


    # Freeze all the model parameters
    if config['pretrained']:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

    # Replace the last fully connected layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config['num_classes'])  # Assuming binary classification

    # Move the model to the appropriate device
    model = model.to(config['device'])

    return model