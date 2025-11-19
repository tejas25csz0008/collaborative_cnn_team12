import os
import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from models.model_v1 import ModelV1
import json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data"


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)


model = ModelV1(num_classes=2).to(DEVICE)
model.load_state_dict(torch.load("models/model_v1.pth", map_location=DEVICE))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="macro")
precision = precision_score(all_labels, all_preds, average="macro")
recall = recall_score(all_labels, all_preds, average="macro")

results = {
    "accuracy": accuracy,
    "f1_score": f1,
    "precision": precision,
    "recall": recall
}

os.makedirs("results", exist_ok=True)
with open("results/test_v1_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("Evaluation complete!")
print(results)
print("Saved to results/test_v1_metrics.json")
