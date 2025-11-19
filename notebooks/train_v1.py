import os
import torch
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from models.model_v1 import ModelV1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

model = ModelV1(num_classes=2).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(2):
    print("Epoch:", epoch+1)
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        y = model(imgs)
        loss = criterion(y, labels)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "models/model_v1.pth")
print("Model saved at models/model_v1.pth")
import os
import torch
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from models.model_v1 import ModelV1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

model = ModelV1(num_classes=2).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(2):
    print("Epoch:", epoch+1)
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        y = model(imgs)
        loss = criterion(y, labels)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "models/model_v1.pth")
print("Model saved at models/model_v1.pth")
