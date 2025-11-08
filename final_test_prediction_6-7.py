import json
import os

# ================= CONFIG =================
BASE = r"C:\Users\Aastha sengar\Desktop\final-deepfake"
INPUT_JSON = os.path.join(BASE, "teamname_all_variants_predictions.json")
OUTPUT_JSON = os.path.join(BASE, "6-7.json")

# ================= LOAD =================
with open(INPUT_JSON, "r") as f:
    data = json.load(f)

print(f"📂 Loaded {len(data)} grouped entries from {INPUT_JSON}")

# ================= PROCESS =================
final_results = []

for item in data:
    index = item["index"]
    variants = item["variants"]

    # average the confidences
    avg_real = sum(v["confidence_real"] for v in variants) / len(variants)
    avg_fake = sum(v["confidence_fake"] for v in variants) / len(variants)

    # choose final prediction based on higher confidence
    prediction = "real" if avg_real >= avg_fake else "fake"

    final_results.append({
        "index": int(index),
        "prediction": prediction
    })

# sort by index
final_results.sort(key=lambda x: x["index"])

# ================= SAVE =================
with open(OUTPUT_JSON, "w") as f:
    json.dump(final_results, f, indent=4)

print(f"✅ Final simplified JSON saved to: {OUTPUT_JSON}")
print(f"🧾 Total entries: {len(final_results)}")
