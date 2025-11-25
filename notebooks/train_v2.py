# models/train_v1.py
import argparse, os, json, time
import torch
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from models.model_v1 import get_model
from utils.metrics import compute_metrics
from tqdm import tqdm

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_t = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_t = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=train_t)
    val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), transform=val_t)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = get_model(num_classes=args.num_classes)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_loader.dataset)
       
        model.eval()
        y_true, y_pred = [], []
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1).cpu().numpy()
                y_pred.extend(preds.tolist())
                y_true.extend(labels.cpu().numpy().tolist())
        val_loss = val_loss / len(val_loader.dataset)
        metrics = compute_metrics(y_true, y_pred)
        
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={metrics['accuracy']:.4f}")


        
        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'models', 'model_v1.pth'))

   
    out_metrics = {
        "dataset": args.dataset_name,
        "best_val_accuracy": best_acc,
        "history": history,
        "final_metrics": metrics,
    }
    os.makedirs(os.path.join(args.out_dir,'results'), exist_ok=True)
    with open(os.path.join(args.out_dir, 'results', 'metrics_v1.json'), 'w') as f:
        json.dump(out_metrics, f, indent=4)
    print("Training complete. Metrics and model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='.')
    parser.add_argument('--dataset_name', type=str, default='Dataset_A')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_classes', type=int, default=2)
    args = parser.parse_args()
    
    os.makedirs(os.path.join(args.out_dir,'models'), exist_ok=True)
    train(args)
