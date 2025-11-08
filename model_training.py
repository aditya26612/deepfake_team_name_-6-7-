"""
train_combined_dataset.py
Fine-tunes EfficientNet-B0 on the combined multi-enhanced Deepfake dataset.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm
import multiprocessing


# ==========================================================
# MAIN TRAINING FUNCTION
# ==========================================================
def main():
    # ==========================================================
    # CONFIGURATION
    # ==========================================================
    BASE_DIR = r"C:\Users\Aastha sengar\Desktop\final-deepfake"
    DATA_DIR = os.path.join(BASE_DIR, "combined_dataset")
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "best_model_efficientnet_b0.pth")

    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4
    VAL_SPLIT = 0.15
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 0  # 🔹 Important fix for Windows multiprocessing issues

    print(f"✅ Using device: {DEVICE.upper()}")

    # ==========================================================
    # TRANSFORMS (Augmentation + Normalization)
    # ==========================================================
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # ==========================================================
    # DATASET SPLIT
    # ==========================================================
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform  # overwrite transform for val

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"📊 Dataset split: {train_size} train / {val_size} val")

    # ==========================================================
    # MODEL SETUP: EfficientNet-B0
    # ==========================================================
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    for param in model.features.parameters():
        param.requires_grad = False

    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_ftrs, 1)
    )

    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # ==========================================================
    # TRAINING FUNCTIONS
    # ==========================================================
    def train_one_epoch(model, loader, criterion, optimizer):
        model.train()
        total_loss, correct = 0, 0

        for images, labels in tqdm(loader, desc="Training", leave=False):
            images, labels = images.to(DEVICE), labels.float().unsqueeze(1).to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).int()
            correct += (preds == labels.int()).sum().item()

        avg_loss = total_loss / len(loader)
        acc = correct / len(loader.dataset)
        return avg_loss, acc

    def validate(model, loader, criterion):
        model.eval()
        total_loss, correct = 0, 0

        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Validating", leave=False):
                images, labels = images.to(DEVICE), labels.float().unsqueeze(1).to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.5).int()
                correct += (preds == labels.int()).sum().item()

        avg_loss = total_loss / len(loader)
        acc = correct / len(loader.dataset)
        return avg_loss, acc

    # ==========================================================
    # TRAINING LOOP
    # ==========================================================
    best_val_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        print(f"\n🚀 Epoch {epoch+1}/{NUM_EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step(val_acc)

        print(f"📈 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"🧪 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ Saved new best model! (Val Acc: {val_acc:.4f})")

    print(f"\n🏁 Training complete. Best Validation Accuracy: {best_val_acc:.4f}")


# ==========================================================
# SAFE ENTRY POINT FOR WINDOWS
# ==========================================================
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
