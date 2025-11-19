import torch
import torch.nn as nn
import torchvision.models as models

class ModelV1(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(ModelV1, self).__init__()
        if pretrained:
            self.base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.base = models.resnet18()

        in_features = self.base.fc.in_features
        self.base.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base(x)
import torch
import torch.nn as nn
import torchvision.models as models

class ModelV1(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(ModelV1, self).__init__()
        if pretrained:
            self.base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.base = models.resnet18()

        in_features = self.base.fc.in_features
        self.base.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base(x)
