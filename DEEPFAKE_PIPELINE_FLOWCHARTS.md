
# 🧩 Deepfake Detection Pipeline — Flowcharts

This document contains standalone flowcharts for the **Deepfake Detection System**.  
It complements the main [README.md](./FINAL_DEEPFAKE_README_GITHUB_SAFE.md) by providing clean, visual explanations.

---

## 🧠 End-to-End Pipeline (Mermaid Diagram)

```mermaid
flowchart TD

Start([Start]) --> A[Raw Dataset: real_cifake_images and fake_cifake_images]
A --> B[Ground Truth JSONs: real_cifake_preds.json and fake_cifake_preds.json]
B --> C[Combine datasets using combined_dataset.py]
C --> D1[Enhancement 1: realesrgan_with_extra.py with RealESRGAN_x4plus.pth]
C --> D2[Enhancement 2: now_enhanced.py (5 augmented variants)]
D1 --> E[Merge enhanced folders using merge_datasets_final.py]
D2 --> E
E --> F[Train model using model_training.py with EfficientNetB0]
F --> G[Save model as 1-1part_model.pth]
G --> H[Prepare and enhance test dataset]
H --> I[Run inference using test_json.py]
I --> J[Aggregate predictions with final_test_prediction_6-7.py]
J --> K[Final output file: 6-7.json]
K --> End([End])
```

---

## 🖼️ Visual Diagram (PNG)

Below is the static image version of the same flowchart for easy embedding or sharing:

![Deepfake Detection Pipeline](364a24c2-66c7-4d13-ab87-de4fd843be56.png)

---

## 🔗 Related Files

- **Main Project Documentation:** [FINAL_DEEPFAKE_README_GITHUB_SAFE.md](./FINAL_DEEPFAKE_README_GITHUB_SAFE.md)
- **Flowchart Image (PNG):** [Deepfake Detection Pipeline Image](./364a24c2-66c7-4d13-ab87-de4fd843be56.png)

---

> You can view this file directly on GitHub to see the rendered Mermaid diagram, or use any Mermaid-compatible Markdown viewer (VS Code, Obsidian, Typora, etc.).
