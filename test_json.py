import os, re, json, torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

# ================= CONFIG =================
BASE = r"C:\Users\Aastha sengar\Desktop\final-deepfake"
MODEL_PATH = os.path.join(BASE, "models", "1-1part_model.pth")

# ✅ Only preprocessed test folders
TEST_FOLDERS = [
    os.path.join(BASE, "test_now_enhanced"),
    os.path.join(BASE, "test_enhanced_realesrgan_with_extra")
]

OUTPUT_JSON = os.path.join(BASE, "teamname_all_variants_predictions.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= TRANSFORM =================
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ================= LOAD MODEL =================
print("🧠 Loading EfficientNet-B0 model...")
model = models.efficientnet_b0(weights=None)
num_f = model.classifier[1].in_features
model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(num_f, 1))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print(f"✅ Model loaded from {MODEL_PATH}")

# ================= IMAGE LISTING =================
def list_images(folder):
    valid_ext = (".png", ".jpg", ".jpeg")
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(valid_ext)
    ])

all_images = []
for folder in TEST_FOLDERS:
    if os.path.exists(folder):
        images = list_images(folder)
        print(f"📂 Found {len(images)} images in {os.path.basename(folder)}")
        all_images.extend(images)
    else:
        print(f"⚠️ Folder not found: {folder}")

print(f"📸 Total test images loaded for prediction: {len(all_images)}\n")

# ================= INFERENCE =================
grouped_predictions = defaultdict(list)

def extract_index(filename):
    """
    Extract the numeric part before '_aug' or '_enhanced' or file extension.
    E.g.:
    129_aug1.jpg -> 129
    129_enhanced.png -> 129
    129.png -> 129
    """
    match = re.match(r"(\d+)", filename)
    return int(match.group(1)) if match else filename

with torch.no_grad():
    for img_path in tqdm(all_images, desc="🔍 Predicting"):
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            continue

        x = tf(img).unsqueeze(0).to(DEVICE)
        out = model(x)
        prob_real = torch.sigmoid(out).item()
        prob_fake = 1 - prob_real
        label = "real" if prob_real >= 0.5 else "fake"

        base = os.path.basename(img_path)
        index = extract_index(base)

        grouped_predictions[index].append({
            "filename": base,
            "prediction": label,
            "confidence_real": round(prob_real, 5),
            "confidence_fake": round(prob_fake, 5)
        })

# ================= BUILD FINAL JSON =================
final_output = []
for index, preds in grouped_predictions.items():
    final_output.append({
        "index": index,
        "variants": sorted(preds, key=lambda x: x["filename"])
    })

final_output.sort(key=lambda x: int(x["index"]) if str(x["index"]).isdigit() else x["index"])

# ================= SAVE =================
with open(OUTPUT_JSON, "w") as f:
    json.dump(final_output, f, indent=4)

print(f"\n✅ Fixed grouping predictions saved to: {OUTPUT_JSON}")
print(f"🧾 Total grouped entries: {len(final_output)}")
