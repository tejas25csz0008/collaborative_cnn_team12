import os
import torch
from torchvision import transforms
from PIL import Image
from models.model_v1 import get_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
TEST_DIR = "/home/tejas/collaborative_cnn_team/collaborative_cnn_team12_/test_data"
MODEL_PATH = "/home/tejas/collaborative_cnn_team/collaborative_cnn_team12_/models/model_v1.pth"
RESULTS_DIR = "/home/tejas/collaborative_cnn_team/collaborative_cnn_team12_/results"
PREDICTIONS_FILE = os.path.join(RESULTS_DIR, "predictions.json")
METRICS_FILE = os.path.join(RESULTS_DIR, "evaluation_metrics.json")

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def load_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)

# Load model
model = get_model(num_classes=2, pretrained=False).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Run inference
all_labels = []
all_preds = []
predictions = {}

label_map = {"cats": 0, "dogs": 1}

for class_folder in os.listdir(TEST_DIR):
    folder_path = os.path.join(TEST_DIR, class_folder)
    if not os.path.isdir(folder_path):
        continue

    true_label = label_map.get(class_folder)
    if true_label is None:
        continue

    for fname in os.listdir(folder_path):
        img_path = os.path.join(folder_path, fname)
        try:
            img = load_image(img_path).to(DEVICE)
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            continue

        with torch.no_grad():
            pred = torch.argmax(model(img), dim=1).item()

        all_labels.append(true_label)
        all_preds.append(pred)
        predictions[img_path] = "cat" if pred == 0 else "dog"

# Compute evaluation metrics
metrics = {
    "accuracy": float(accuracy_score(all_labels, all_preds)),
    "precision": float(precision_score(all_labels, all_preds, average='macro')),
    "recall": float(recall_score(all_labels, all_preds, average='macro')),
    "f1_score": float(f1_score(all_labels, all_preds, average='macro'))
}

# Save predictions
os.makedirs(RESULTS_DIR, exist_ok=True)
with open(PREDICTIONS_FILE, "w") as f:
    json.dump(predictions, f, indent=4)

# Save evaluation metrics
with open(METRICS_FILE, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Predictions saved to: {PREDICTIONS_FILE}")
print(f"Evaluation metrics saved to: {METRICS_FILE}")
print("Done!")
