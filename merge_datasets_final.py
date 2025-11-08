import os
import json
import shutil
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = r"C:\Users\Aastha sengar\Desktop\final-deepfake"

# Input folders (original mixed ones)
REAL_CIFAKE_FOLDER = os.path.join(BASE_DIR, "real_cifake_images")
FAKE_CIFAKE_FOLDER = os.path.join(BASE_DIR, "fake_cifake_images")

# JSON ground truth files
REAL_JSON = os.path.join(BASE_DIR, "real_cifake_preds.json")
FAKE_JSON = os.path.join(BASE_DIR, "fake_cifake_preds.json")

# Output (cleanly separated dataset)
OUTPUT_DIR = os.path.join(BASE_DIR, "sorted_dataset")
REAL_OUT = os.path.join(OUTPUT_DIR, "real")
FAKE_OUT = os.path.join(OUTPUT_DIR, "fake")

os.makedirs(REAL_OUT, exist_ok=True)
os.makedirs(FAKE_OUT, exist_ok=True)


# ==========================================
# HELPER FUNCTION
# ==========================================
def find_image(index, folder):
    """Find an image by index (1.png, 2.jpg, etc.)."""
    for ext in [".png", ".jpg", ".jpeg"]:
        path = os.path.join(folder, f"{index}{ext}")
        if os.path.exists(path):
            return path
    return None


def process_json(json_path, src_folder):
    """Read JSON file and move images into real/fake folders."""
    with open(json_path, "r") as f:
        data = json.load(f)

    moved = 0
    missing = 0
    print(f"\n📖 Processing {os.path.basename(json_path)} ({len(data)} entries)")

    for entry in tqdm(data):
        index = entry.get("index")
        label = entry.get("prediction", "").lower().strip()

        src = find_image(index, src_folder)
        if not src:
            missing += 1
            continue

        dest_folder = REAL_OUT if label == "real" else FAKE_OUT
        shutil.copy(src, os.path.join(dest_folder, os.path.basename(src)))
        moved += 1

    print(f"✅ Done: {moved} images moved from {os.path.basename(src_folder)}")
    if missing > 0:
        print(f"⚠️ Missing {missing} files (check index or extension)")
    return moved, missing


# ==========================================
# MAIN SCRIPT
# ==========================================
if __name__ == "__main__":
    print("🚀 Starting separation of real/fake images using JSON ground truth...\n")

    moved1, miss1 = process_json(REAL_JSON, REAL_CIFAKE_FOLDER)
    moved2, miss2 = process_json(FAKE_JSON, FAKE_CIFAKE_FOLDER)

    total_moved = moved1 + moved2
    total_missing = miss1 + miss2

    print("\n================ SUMMARY ================")
    print(f"📂 Total images moved: {total_moved}")
    print(f"⚠️ Total missing: {total_missing}")
    print(f"✅ Real images → {REAL_OUT}")
    print(f"✅ Fake images → {FAKE_OUT}")
    print("=========================================\n")
