import os, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ================= CONFIG =================
BASE = r"C:\Users\Aastha sengar\Desktop\final-deepfake"
DATA = os.path.join(BASE, "combined_dataset_withreal_and_fake_ground_truth")
SAVE = os.path.join(BASE, "models", "1-1part_model.pth")
os.makedirs(os.path.dirname(SAVE), exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#if cuda is not available, use cpu printf message what is being used
print(f"✅ Using device: {DEVICE.upper()}")
BATCH, EPOCHS, LR = 32, 25, 1e-4

# ================= TRANSFORMS (NO AUGMENTATION) =================
basic_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ================= ORDERED SPLIT =================
print("📦 Loading dataset (85% train / 15% val, ordered)...")
full_dataset = datasets.ImageFolder(DATA, transform=basic_tf)
full_dataset.samples.sort(key=lambda x: x[0])
n = len(full_dataset)
split = int(n * 0.85)
train_dataset = Subset(full_dataset, range(0, split))
val_dataset   = Subset(full_dataset, range(split, n))
train_loader = DataLoader(train_dataset, BATCH, shuffle=True)
val_loader   = DataLoader(val_dataset, BATCH, shuffle=False)
print(f"✅ Loaded: {len(train_dataset)} train | {len(val_dataset)} val")

# ================= MODEL =================
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

# Only unfreeze last blocks (6 & 7)
for name, p in model.features.named_parameters():
    p.requires_grad = any(x in name for x in ["6", "7"])

num_f = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_f, 1)
)
model = model.to(DEVICE)

# ================= OPTIMIZER, LOSS, SCHEDULER =================
crit = nn.BCEWithLogitsLoss()
opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = optim.lr_scheduler.ReduceLROnPlateau(opt, "max", patience=3)

# ================= TRAIN LOOP =================
def run(loader, train=True):
    model.train(train)
    tot_loss, correct, total = 0, 0, 0
    desc = "🟢 Train" if train else "🔵 Val"
    
    for x, y in tqdm(loader, desc=desc, leave=False):
        x, y = x.to(DEVICE), y.float().unsqueeze(1).to(DEVICE)
        with torch.set_grad_enabled(train):
            out = model(x)
            loss = crit(out, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
        tot_loss += loss.item()
        pred = (torch.sigmoid(out) > 0.5).int()
        correct += (pred == y.int()).sum().item()
        total += y.size(0)
    return tot_loss / len(loader), correct / total

# ================= TRAINING =================
best = 0
for e in range(EPOCHS):
    print(f"\n🌍 Epoch {e+1}/{EPOCHS}\n" + "-"*50)
    tl, ta = run(train_loader, True)
    vl, va = run(val_loader, False)
    sched.step(va)
    print(f"📈 Train Loss={tl:.4f} Acc={ta*100:.2f}% | Val Loss={vl:.4f} Acc={va*100:.2f}%")
    if va > best:
        best = va
        torch.save(model.state_dict(), SAVE)
        print(f"💾 New best saved (Val {va*100:.2f}%)")

print(f"\n🏁 Done — Best Val={best*100:.2f}% | Saved→ {SAVE}")

