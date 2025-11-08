import os
import shutil
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = r"C:\Users\Aastha sengar\Desktop\final-deepfake"

# Input folders (real and fake)
REAL_FOLDERS = [
    
    r"real_now_enhanced",
    r"real_enhanced_realesrgan_with_extra"
]

FAKE_FOLDERS = [
    
    r"fake_now_enhanced",
    r"fake_enhanced_realesrgan_with_extra"
]

# Output combined dataset
OUTPUT_DIR = os.path.join(BASE_DIR, "combined_dataset_withreal_and_fake_ground_truth")
REAL_OUT = os.path.join(OUTPUT_DIR, "real")
FAKE_OUT = os.path.join(OUTPUT_DIR, "fake")

os.makedirs(REAL_OUT, exist_ok=True)
os.makedirs(FAKE_OUT, exist_ok=True)

# ==========================================
# HELPER FUNCTION
# ==========================================
def copy_images(src_folder, dest_folder):
    """Copy all image files from one folder to another with prefixed names."""
    count = 0
    folder_prefix = os.path.basename(src_folder).replace("-", "_").replace(" ", "_")
    for file in tqdm(os.listdir(src_folder), desc=f"Copying {os.path.basename(src_folder)}"):
        src_path = os.path.join(src_folder, file)
        if not os.path.isfile(src_path):
            continue
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            new_name = f"{folder_prefix}_{file}"  # e.g. real_now_enhanced_1_aug1.jpg
            dest_path = os.path.join(dest_folder, new_name)
            shutil.copy2(src_path, dest_path)
            count += 1
    return count

# ==========================================
# MAIN LOGIC
# ==========================================
if __name__ == "__main__":
    print("🚀 Combining real and fake folders into combined_dataset...\n")

    total_real, total_fake = 0, 0

    for folder in REAL_FOLDERS:
        src = os.path.join(BASE_DIR, folder)
        if os.path.exists(src):
            total_real += copy_images(src, REAL_OUT)
        else:
            print(f"⚠️ Skipped missing folder: {src}")

    for folder in FAKE_FOLDERS:
        src = os.path.join(BASE_DIR, folder)
        if os.path.exists(src):
            total_fake += copy_images(src, FAKE_OUT)
        else:
            print(f"⚠️ Skipped missing folder: {src}")

    print("\n================ SUMMARY ================")
    print(f"✅ Total REAL images copied: {total_real}")
    print(f"✅ Total FAKE images copied: {total_fake}")
    print(f"📂 Combined dataset path: {OUTPUT_DIR}")
    print("=========================================\n")
