"""
enhance_and_preprocess.py
Enhanced preprocessing pipeline for Deepfake Detection Challenge.
Combines Real-ESRGAN super-resolution, adaptive post-processing,
and domain-specific augmentations.
"""

import argparse
import os
import sys
import cv2
import numpy as np
import torch
from tqdm import tqdm

# ---- FIX for torchvision >=0.15 ----
# (prevents "No module named 'torchvision.transforms.functional_tensor'")
try:
    import torchvision.transforms.functional as F
    if not hasattr(F, "rgb_to_grayscale"):
        import torchvision.transforms.functional_tensor as ft
        F.rgb_to_grayscale = ft.rgb_to_grayscale
except Exception:
    pass
# ------------------------------------

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


# ---- DEFAULT CONFIG ----
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "RealESRGAN_x4plus.pth")
DEFAULT_INPUT = os.path.join(HERE, "sorted_dataset", "real")
DEFAULT_OUTPUT = os.path.join(HERE, "fake_now_enhanced")


# ---- ARGUMENT PARSING ----
def parse_args():
    p = argparse.ArgumentParser(description="Enhance images with Real-ESRGAN and postprocess/augment")
    p.add_argument("--input", "-i", default=DEFAULT_INPUT, help="Input folder with images")
    p.add_argument("--output", "-o", default=None, help="Output folder for enhanced images (auto-derived if not set)")
    p.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Path to RealESRGAN model (.pth)")
    p.add_argument("--no-augment", dest="augment", action="store_false", help="Disable augmentation step")
    return p.parse_args()


# ---- MAIN SETUP ----
def setup_paths(args):
    """Handle input/output/model paths."""
    input_folder = args.input
    if not os.path.isdir(input_folder):
        print(f"❌ Input folder not found: {input_folder}")
        sys.exit(1)

    if args.output:
        output_folder = args.output
    else:
        base = os.path.basename(os.path.normpath(input_folder)) or "enhanced"
        output_folder = os.path.join(HERE, f"{base}_now_enhanced")

    os.makedirs(output_folder, exist_ok=True)

    model_path = args.model
    if not os.path.isfile(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please download 'RealESRGAN_x4plus.pth' and place it correctly.")
        sys.exit(1)

    return input_folder, output_folder, model_path


# ---- IMAGE ENHANCEMENT UTILS ----
def auto_sharpen(img):
    """Adaptive sharpening based on variance."""
    variance = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    strength = np.clip(variance / 100.0, 0.5, 2.0)
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)


def clahe_enhance(img):
    """Contrast Limited Adaptive Histogram Equalization."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def detect_and_crop_face(img):
    """Detect and crop the largest face; fallback to original."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        margin = int(0.2 * w)
        y1, y2 = max(0, y - margin), min(img.shape[0], y + h + margin)
        x1, x2 = max(0, x - margin), min(img.shape[1], x + w + margin)
        cropped = img[y1:y2, x1:x2]
        return cropped if cropped.size > 0 else img
    return img


def augment(img):
    """Create a set of augmented versions of the image."""
    return [
        cv2.flip(img, 1),
        cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
        cv2.GaussianBlur(img, (5, 5), 1),
        cv2.convertScaleAbs(img, alpha=1.1, beta=10),
    ]


# ---- CORE PROCESSING ----
def process_images(input_folder, output_folder, upsampler, augment_flag=True):
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not files:
        print(f"⚠️ No images found in {input_folder}")
        return

    for file in tqdm(files, desc="Enhancing & preprocessing"):
        path = os.path.join(input_folder, file)
        try:
            img = cv2.imread(path)
            if img is None:
                print(f"⚠️ Skipping unreadable file: {file}")
                continue

            # Face crop
            img = detect_and_crop_face(img)

            # Real-ESRGAN enhancement
            enhanced, _ = upsampler.enhance(img, outscale=4)

            # Postprocess: sharpen, contrast enhance, denoise
            sharpened = auto_sharpen(enhanced)
            contrast = clahe_enhance(sharpened)
            final = cv2.fastNlMeansDenoisingColored(contrast, None, 10, 10, 7, 21)

            # Save enhanced
            out_path = os.path.join(output_folder, file)
            cv2.imwrite(out_path, final)

            # Augmentations
            if augment_flag:
                for i, aug_img in enumerate(augment(final)):
                    aug_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_aug{i}.jpg")
                    cv2.imwrite(aug_path, aug_img)

        except Exception as e:
            print(f"❌ Error processing {file}: {e}")
            continue

    print("\n✅ Enhanced preprocessing complete!")


# ---- MAIN ENTRY POINT ----
if __name__ == "__main__":
    args = parse_args()
    INPUT_FOLDER, OUTPUT_FOLDER, MODEL_PATH = setup_paths(args)

    print(f"\n📂 Input: {INPUT_FOLDER}")
    print(f"📁 Output: {OUTPUT_FOLDER}")
    print(f"🧠 Model: {MODEL_PATH}")
    print(f"✨ Augmentation: {args.augment}\n")

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                    num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=MODEL_PATH,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available()
    )

    process_images(INPUT_FOLDER, OUTPUT_FOLDER, upsampler, augment_flag=args.augment)
