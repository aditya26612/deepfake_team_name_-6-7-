
# 🧠 Deepfake Detection System — End‑to‑End (Open Source)

**Author:** Aditya Singh Senger  
**Goal:** Detect *real vs. fake* images using a robust pipeline that cleans raw CIFAKE data, enhances images with a **pre-trained Real-ESRGAN model (`RealESRGAN_x4plus.pth`)**, trains an EfficientNet-B0 classifier, and aggregates predictions over multiple enhanced variants.

> From pixels to truth — detecting deception one frame at a time.

---

## 🧭 Project Architecture Overview

![Deepfake Pipeline Diagram](fbe99570-cb8f-4048-bdef-445c3f635487.png)

---

## 📂 Repository Structure (with Folder Relationships)

```
Deepfake-Detection/
│
├── combined_dataset_withreal_and_fake_ground_truth/
│   ├── real/                               # Real images separated using combined_dataset.py
│   └── fake/                               # Fake images separated using combined_dataset.py
│
├── fake_cifake_images/                     # Raw unprocessed fake CIFAKE dataset
├── real_cifake_images/                     # Raw unprocessed real CIFAKE dataset
│
├── fake_cifake_preds.json                  # Ground truth JSON for fake CIFAKE images
├── real_cifake_preds.json                  # Ground truth JSON for real CIFAKE images
│
├── fake_enhanced_realesrgan_with_extra/    # Single enhanced (clean) fake images using RealESRGAN_x4plus.pth
├── fake_now_enhanced/                      # Multiple augmented enhanced fake images (5 variants each)
├── real_enhanced_realesrgan_with_extra/    # Single enhanced (clean) real images using RealESRGAN_x4plus.pth
├── real_now_enhanced/                      # Multiple augmented enhanced real images (5 variants each)
│
├── sorted_dataset/                         # Final structured dataset for training and validation
│   ├── real/                               # Enhanced + merged real images
│   ├── fake/                               # Enhanced + merged fake images
│   └── test/                               # Enhanced + merged test images
│
├── test/                                   # Raw test dataset before preprocessing
├── test_enhanced_realesrgan_with_extra/    # Single enhanced test images using RealESRGAN_x4plus.pth
├── test_now_enhanced/                      # Multiple augmented enhanced test images
│
├── models/
│   ├── 1-1part_model.pth                   # Pre-trained EfficientNet-B0 model checkpoint (fine-tuned)
│   └── RealESRGAN_x4plus.pth               # Pre-trained Real-ESRGAN model used for image enhancement
│
├── merge_datasets_final.py                 # Merges all enhanced folders into final sorted_dataset
├── combined_dataset.py                     # Creates ground-truth-based combined dataset (real/fake)
│
├── .._enhanced_realesrgan_with_extra.py    # Real-ESRGAN enhancement (1x clean enhancement per image)
├── .._now_enhanced.py                      # Real-ESRGAN + augmentations (5x per image)
│
├── model_training.py                       # EfficientNet-B0 model training script
├── test_json.py                            # Inference script: outputs JSON with grouped predictions
├── final_test_prediction_6-7.py            # Averages predictions → final 6-7.json output
│
├── check_cuda.py                           # Verifies CUDA / GPU setup
├── requirements.txt                        # Required dependencies
├── .gitattributes                          # Enables Git LFS for large file tracking
└── README.md                               # This file
```

---

## 🧠 End‑to‑End Pipeline Flow

```mermaid
flowchart TD
A[Raw CIFAKE Data<br/>real_cifake_images + fake_cifake_images] --> B[Ground Truth Separation<br/>real_cifake_preds.json + fake_cifake_preds.json]
B --> C[Combine Datasets<br/>combined_dataset.py → combined_dataset_withreal_and_fake_ground_truth/]
C --> D1[Enhancement 1× (Clean)<br/>.._enhanced_realesrgan_with_extra.py<br/>Using pre-trained RealESRGAN_x4plus.pth]
C --> D2[Enhancement 5× (Augmented)<br/>.._now_enhanced.py<br/>Rotation, Flip, Brightness]
D1 --> E[Merge Enhanced Data<br/>merge_datasets_final.py → sorted_dataset/real + fake]
D2 --> E
E --> F[Train EfficientNet‑B0<br/>model_training.py<br/>Frozen backbone + fine-tuned classifier]
F --> G[Save Trained Model<br/>models/best_model_efficientnet_b0.pth]
G --> H[Prepare Test Set (same enhancements)]
H --> I[Inference<br/>test_json.py → teamname_all_variants_predictions.json]
I --> J[Aggregate Predictions<br/>final_test_prediction_6-7.py → 6-7.json]
```

---

## ⚙️ Model Details

| Parameter | Configuration |
|------------|----------------|
| **Backbone** | EfficientNet‑B0 (Pre-trained on ImageNet) |
| **Enhancer** | Real‑ESRGAN (`RealESRGAN_x4plus.pth`) |
| **Fine‑tuning** | Frozen backbone; classifier head trained |
| **Split** | Ordered 85% / 15% (variant-consistent) |
| **Loss Function** | BCEWithLogitsLoss |
| **Optimizer** | AdamW (LR=1e‑4, weight_decay=1e‑5) |
| **Scheduler** | ReduceLROnPlateau (factor=0.5, patience=3) |
| **Batch Size** | 32 |
| **Image Size** | 224×224 |
| **Framework** | PyTorch + Torchvision |
| **Output** | Binary Classification — *Real* / *Fake* |

---

## 🧩 Reproducible Steps

### 1️⃣ Create Ground‑Truth Dataset
```bash
python combined_dataset.py
```

### 2️⃣ Enhance Real & Fake Folders
```bash
python .._enhanced_realesrgan_with_extra.py
python .._now_enhanced.py
```

### 3️⃣ Merge Enhanced Sets
```bash
python merge_datasets_final.py
```

### 4️⃣ Train Model
```bash
python model_training.py
```

### 5️⃣ Preprocess Test Data
```bash
python .._enhanced_realesrgan_with_extra.py
python .._now_enhanced.py
python merge_datasets_final.py
```

### 6️⃣ Run Inference
```bash
python test_json.py
```

### 7️⃣ Aggregate Predictions
```bash
python final_test_prediction_6-7.py
```

---

## 🧾 JSON Outputs Example

**Grouped Predictions (teamname_all_variants_predictions.json):**
```json
{
  "index": 2,
  "variants": [
    {"filename": "2.png", "prediction": "real", "confidence_real": 0.99999, "confidence_fake": 0.00001},
    {"filename": "2_aug1.jpg", "prediction": "real", "confidence_real": 0.99805, "confidence_fake": 0.00195}
  ]
}
```

**Final Output (6-7.json):**
```json
[
  {"index": 1, "prediction": "fake"},
  {"index": 2, "prediction": "real"}
]
```

---

## ⚡ Requirements & Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

Then install PyTorch:
```bash
# CUDA build
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# or CPU build
pip install torch torchvision torchaudio
```

Verify CUDA setup:
```bash
python check_cuda.py
```

---

## 🧰 Git LFS Setup

For managing large models and datasets:
```bash
git lfs install
git lfs track "*.pth" "*.zip" "*.png" "*.jpg" "*.jpeg"
git add .gitattributes
git add .
git commit -m "Add deepfake detection pipeline with LFS"
git push -u origin main
```

---

## 🧑‍💻 Author

**Aditya Singh Senger**  
Deepfake Detection • Computer Vision • AI Research  
📧 gokusengar666@gmail.com
🔗 [GitHub Repository](https://github.com/aditya26612/deepfake_team_name_-6-7-)

---



---


---

---

---

## 🧩 Deepfake Detection Pipeline — Visual Overview

The following diagram shows the complete Deepfake Detection pipeline in a simple, GitHub-renderable flow.

```mermaid
flowchart TD

A1([Start]) --> A2[Raw Dataset: real and fake CIFAKE images]
A2 --> A3[Ground Truth JSONs: real_cifake_preds and fake_cifake_preds]
A3 --> A4[Combine Datasets using combined_dataset.py]

A4 --> B1[Enhancement 1 using realesrgan_with_extra.py (clean images)]
A4 --> B2[Enhancement 2 using now_enhanced.py (five augmented variants)]
B1 --> C1[Merge Enhanced Folders using merge_datasets_final.py]
B2 --> C1

C1 --> D1[Train EfficientNetB0 using model_training.py]
D1 --> D2[Save Model as 1-1part_model.pth]

D2 --> E1[Test Data Preparation]
E1 --> E2[Enhance Test Data using same scripts]
E2 --> E3[Merge Enhanced Test Data]
E3 --> E4[Run Inference with test_json.py]
E4 --> F1[Aggregate Predictions with final_test_prediction_6-7.py]
F1 --> F2[Final Output: 6-7.json]
F2 --> G1([End])

classDef data fill:#D6EAF8,stroke:#1B4F72,color:#1B2631,font-weight:bold
classDef process fill:#D5F5E3,stroke:#145A32,color:#0B5345,font-weight:bold
classDef model fill:#FADBD8,stroke:#7B241C,color:#641E16,font-weight:bold
classDef output fill:#FCF3CF,stroke:#7D6608,color:#7E5109,font-weight:bold
classDef end fill:#D2B4DE,stroke:#4A235A,color:#512E5F,font-weight:bold

class A2,A3,A4,E1 data
class B1,B2,C1,E2,E3 process
class D1,D2 model
class E4,F1,F2 output
class A1,G1 end
```

### 🖼️ Visual Pipeline (Image Backup)
If Mermaid doesn’t render, here’s a PNG version of the same pipeline:

![Deepfake Detection Pipeline](364a24c2-66c7-4d13-ab87-de4fd843be56.png)

---

Below is the full end-to-end workflow of the Deepfake Detection System, showing how data flows from raw CIFAKE images to the final JSON prediction output.

### 📊 Mermaid Flowchart (GitHub-Compatible)
```mermaid
flowchart TD

A1([Start]) --> A2[Raw Dataset: real_cifake_images + fake_cifake_images]
A2 --> A3[Ground Truth JSONs: real_cifake_preds.json + fake_cifake_preds.json]
A3 --> A4[Combine Datasets (combined_dataset.py) -> combined_dataset_withreal_and_fake_ground_truth]

A4 --> B1[Enhancement 1 (Clean) using .._enhanced_realesrgan_with_extra.py and RealESRGAN_x4plus.pth + CLAHE + Denoise]
A4 --> B2[Enhancement 2 (Augmented) using .._now_enhanced.py with rotations, flips, brightness variants]

B1 --> C1[Merge Enhanced Datasets (merge_datasets_final.py) -> sorted_dataset/real + fake]
B2 --> C1

C1 --> D1[Model Training (model_training.py) -> EfficientNet-B0 fine-tuning (Layers 6 and 7 trainable)]
D1 --> D2[Saved Model (1-1part_model.pth)]

D2 --> E1[Test Dataset (test folder)]
E1 --> E2[Enhance Test Dataset using both enhancement scripts]
E2 --> E3[Merge Test Datasets (merge_datasets_final.py)]
E3 --> E4[Inference (test_json.py) -> teamname_all_variants_predictions.json]
E4 --> F1[Aggregate Predictions (final_test_prediction_6-7.py)]
F1 --> F2[Final Output: 6-7.json]

F2 --> G1([End])

classDef data fill:#D6EAF8,stroke:#1B4F72,color:#1B2631,font-weight:bold
classDef process fill:#D5F5E3,stroke:#145A32,color:#0B5345,font-weight:bold
classDef model fill:#FADBD8,stroke:#7B241C,color:#641E16,font-weight:bold
classDef output fill:#FCF3CF,stroke:#7D6608,color:#7E5109,font-weight:bold
classDef end fill:#D2B4DE,stroke:#4A235A,color:#512E5F,font-weight:bold

class A2,A3,A4,E1 data
class B1,B2,C1,E2,E3 process
class D1,D2 model
class E4,F1,F2 output
class A1,G1 end
```

---

### 🖼️ Visual Pipeline Diagram (PNG Backup)
If GitHub rendering fails or you prefer an image view, here’s the same flow as a PNG diagram:

![Deepfake Detection Pipeline](364a24c2-66c7-4d13-ab87-de4fd843be56.png)

---

The following flowchart illustrates the complete end-to-end workflow of this project — from dataset preparation and enhancement to model training and prediction aggregation.

```mermaid
flowchart TD

%% DATASET
A1([Start]) --> A2[📦 Raw Dataset<br>real_cifake_images + fake_cifake_images]
A2 --> A3[📑 Ground Truth JSONs<br>real_cifake_preds.json + fake_cifake_preds.json]
A3 --> A4[🗂️ Combine Datasets<br>combined_dataset.py<br>→ combined_dataset_withreal_and_fake_ground_truth/real + fake]

%% ENHANCEMENT STAGE
A4 --> B1[🔍 Enhancement 1<br>.._enhanced_realesrgan_with_extra.py<br>→ RealESRGAN_x4plus.pth + CLAHE + Sharpen + Denoise<br>Generates 1 enhanced image per input]
A4 --> B2[🎨 Enhancement 2<br>.._now_enhanced.py<br>→ RealESRGAN_x4plus.pth + Augmentations<br>Generates 5 enhanced variants per input]
B1 --> C1[🧩 Merge Enhanced Folders<br>merge_datasets_final.py<br>→ sorted_dataset/real + fake]
B2 --> C1

%% TRAINING
C1 --> D1[🧠 Model Training<br>model_training.py<br>EfficientNet-B0 (Layers 6 & 7 trainable)]
D1 --> D2[(💾 1-1part_model.pth<br>Saved Model in /models)]

%% TESTING PIPELINE
D2 --> E1[🧾 Test Dataset<br>test/ folder (real + fake test images)]
E1 --> E2[🧪 Enhance Test Folders<br>.._enhanced_realesrgan_with_extra.py + .._now_enhanced.py]
E2 --> E3[🔗 Merge Test Enhancements<br>merge_datasets_final.py]
E3 --> E4[🤖 Run Inference<br>test_json.py<br>→ teamname_all_variants_predictions.json (confidence scores)]

%% AGGREGATION
E4 --> F1[📊 Aggregate Results<br>final_test_prediction_6-7.py<br>Average confidence per image]
F1 --> F2[✅ Final Output<br>6-7.json<br>Index + Prediction (real/fake)]
F2 --> G1([🏁 End])

%% STYLING
classDef data fill:#D6EAF8,stroke:#1B4F72,color:#1B2631,font-weight:bold
classDef process fill:#D5F5E3,stroke:#145A32,color:#0B5345,font-weight:bold
classDef model fill:#FADBD8,stroke:#7B241C,color:#641E16,font-weight:bold
classDef output fill:#FCF3CF,stroke:#7D6608,color:#7E5109,font-weight:bold
classDef end fill:#D2B4DE,stroke:#4A235A,color:#512E5F,font-weight:bold

class A2,A3,A4,E1 data
class B1,B2,C1,E2,E3 process
class D1,D2 model
class E4,F1,F2 output
class A1,G1 end
```
---

## 🎓 Dataset Information

The dataset used in this project was provided by the **Deepfake Hackathon organized by IIIT Bengaluru**.  
It contains both **real** and **fake CIFAKE images**, with accompanying JSON annotations that indicate the ground truth for each image.  
These were used to create structured folders via `combined_dataset.py` for downstream enhancement and training.

---

## 🧠 Model Training Code Overview (`model_training.py`)

This script trains the **EfficientNet-B0** model on the enhanced CIFAKE dataset with frozen lower layers, fine-tuned top blocks (6 & 7), and a custom binary classifier head.

### 🔧 Key Features
- **Automatic device selection:** Uses CUDA if available, else CPU.
- **No training augmentation:** Works on clean enhanced images only.
- **Ordered 85/15 split:** Keeps image variants grouped consistently.
- **Partial fine-tuning:** Only EfficientNet layers **6 & 7** are trainable.
- **Optimizer:** AdamW with weight decay (1e-4).
- **Scheduler:** ReduceLROnPlateau (based on validation accuracy).
- **Loss:** BCEWithLogitsLoss.

### 🧩 Training Process

1️⃣ **Dataset Loading and Normalization**
```python
basic_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
full_dataset = datasets.ImageFolder(DATA, transform=basic_tf)
```

2️⃣ **Ordered Split**
```python
full_dataset.samples.sort(key=lambda x: x[0])
split = int(len(full_dataset) * 0.85)
train_dataset = Subset(full_dataset, range(0, split))
val_dataset   = Subset(full_dataset, range(split, len(full_dataset)))
```

3️⃣ **Model Definition**
```python
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
for name, p in model.features.named_parameters():
    p.requires_grad = any(x in name for x in ["6", "7"])

num_f = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_f, 1)
)
```

4️⃣ **Training Loop**
```python
for e in range(EPOCHS):
    tl, ta = run(train_loader, True)
    vl, va = run(val_loader, False)
    sched.step(va)
    if va > best:
        torch.save(model.state_dict(), SAVE)
```

5️⃣ **Result Example**
```
🌍 Epoch 8/25
📈 Train Loss=0.1321 Acc=97.54% | Val Loss=0.1853 Acc=95.67%
💾 New best saved (Val 95.67%)
🏁 Done — Best Val=95.67% | Saved→ models/1-1part_model.pth
```

### 📊 Summary of Model Design Choices

| Component | Purpose |
|------------|----------|
| EfficientNet‑B0 | Lightweight, accurate architecture pretrained on ImageNet |
| Frozen Layers | Preserve low-level general features |
| Trainable Layers (6 & 7) | Learn deepfake-specific textures and semantics |
| AdamW + ReduceLROnPlateau | Stable, adaptive optimization |
| BCEWithLogitsLoss | Ideal for binary classification (real/fake) |

The final model is saved at:
```
models/1-1part_model.pth
```
and used later for prediction by `test_json.py` and aggregation by `final_test_prediction_6-7.py`.

---

## 🪪 License

**MIT License** — Free to use, modify, and distribute with attribution.

---

## 🙏 Acknowledgments

- **Real‑ESRGAN** team for pre-trained `RealESRGAN_x4plus.pth` super-resolution model.  
- **PyTorch** for model training framework.  
- **CIFAKE Dataset** authors for the benchmark dataset.  

---

> 🧩 *"From pixels to truth — detecting deception one frame at a time."*
