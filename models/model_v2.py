# models/model_v1.py
import torch
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=2, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model