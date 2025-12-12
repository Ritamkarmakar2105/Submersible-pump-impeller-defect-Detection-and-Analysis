from pathlib import Path
from typing import List
import json

from PIL import Image
from ultralytics import YOLO
import ollama
import torch

# ===================== CONFIG =====================

# Base project folder
BASE_DIR = Path(r"D:\submersible pump impeller defect Detection analysis")

# ✅ Path to YOUR finetuned YOLO model
YOLO_MODEL_PATH = Path(
    r"D:\submersible pump impeller defect Detection analysis\Finetuned Model.pt"
)

# ✅ Folder of images to inspect
IMAGES_FOLDER = Path(
    r"D:\submersible pump impeller defect Detection analysis\casting_data\casting_data\train\small dataset for testing (output)"
)

# Ollama model name -> check `ollama list`
LLAVA_MODEL_NAME = "llava:7b"   # or "llava" if that's what you see

# Root output folder
OUTPUT_DIR = BASE_DIR / "inspector_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# YOLO detection params
CONF_THRES = 0.20      # lower = more recall
IOU_THRES = 0.5
IMG_SIZE = 512

# High recall but skip garbage:
MIN_LLAVA_CONF = 0.10      # send almost all reasonably-confident boxes
MIN_BOX_SIZE_FRAC = 0.002  # only skip tiny specks (<0.2% of image area)

# ===================== DEVICE SELECTION =====================

if torch.cuda.is_available():
    DEVICE = 0  # GPU 0
    print("✅ Using GPU (device=0)")
else:
    DEVICE = "cpu"
    print("⚠️ CUDA not available, falling back to CPU")


# ===================== HELPERS =====================

def explain_defect_with_llava(crop_path: Path, bbox_norm, conf: float) -> str:
    prompt = f"""
You are an industrial inspector for submersible pump impellers.

You will see a small cropped image around a suspected defect region.

Bounding box (normalized xywh): {bbox_norm}
Detection confidence (0-1): {conf:.2f}

Answer VERY BRIEFLY in exactly this format, with one short sentence per field:

defect: <what is the defect?>
cause: <how did this defect likely occur?>
repair: <how can this be fixed or handled?>

Do NOT add extra text. Do NOT explain anything outside these three fields.
""".strip()

    response = ollama.chat(
        model=LLAVA_MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(crop_path)],
            }
        ],
    )
    return response["message"]["content"]


def parse_llava_explanation(text: str) -> dict:
    out = {"raw": text, "defect": "", "cause": "", "repair": ""}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("defect:"):
            out["defect"] = line.split(":", 1)[1].strip()
        elif lower.startswith("cause:"):
            out["cause"] = line.split(":", 1)[1].strip()
        elif lower.startswith("repair:"):
            out["repair"] = line.split(":", 1)[1].strip()
    return out


# ===================== CORE INSPECTION =====================

def inspect_image(model: YOLO, image_path: Path) -> Path:
    """
    High-recall version:
    - run YOLO once (model is passed in)
    - send almost all detected defects to LLaVA
    - save annotated full image + inspection.txt in a per-image folder
    """
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    print(f"\n🖼️ Inspecting: {image_path.name}")

    results = model.predict(
        source=str(image_path),
        conf=CONF_THRES,   # main YOLO threshold
        iou=IOU_THRES,
        imgsz=IMG_SIZE,
        device=DEVICE,
        save=False,
        verbose=False,
    )[0]

    h, w = results.orig_shape
    orig_img = Image.open(image_path).convert("RGB")

    # Folder for this image
    img_out_dir = OUTPUT_DIR / image_path.stem
    crops_dir = img_out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Save annotated full image from YOLO
    annotated_np = results.plot()
    annotated_img = Image.fromarray(annotated_np[..., ::-1])  # BGR -> RGB
    annotated_path = img_out_dir / "full.jpg"
    annotated_img.save(annotated_path)

    defect_summaries = []

    print(f"  YOLO detections total: {len(results.boxes)}")

    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls[0])

        # assuming class 0 = defect
        if cls_id != 0:
            continue

        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        box_w = x2 - x1
        box_h = y2 - y1
        box_frac = (box_w * box_h) / (w * h + 1e-9)

        # 🔸 very gentle filters now

        # skip ultra-tiny specks only
        if box_frac < MIN_BOX_SIZE_FRAC:
            print(f"    skipping defect {i}: too small (area frac={box_frac:.4f})")
            continue

        # skip super-low confidence only
        if conf < MIN_LLAVA_CONF:
            print(f"    skipping defect {i}: low conf={conf:.2f}")
            continue

        print(f"  🔍 sending defect {i} to LLaVA: conf={conf:.2f}, area_frac={box_frac:.3f}")

        # crop region for LLaVA
        crop = orig_img.crop((x1, y1, x2, y2))
        crop_path = crops_dir / f"defect_{i}.jpg"
        crop.save(crop_path)

        bbox_norm = [
            ((x1 + x2) / 2) / w,
            ((y1 + y2) / 2) / h,
            box_w / w,
            box_h / h,
        ]

        raw_text = explain_defect_with_llava(crop_path, bbox_norm, conf)
        parsed = parse_llava_explanation(raw_text)

        defect_summaries.append(
            {
                "id": i,
                "confidence": conf,
                "bbox_xyxy": [x1, y1, x2, y2],
                "defect": parsed["defect"],
                "cause": parsed["cause"],
                "repair": parsed["repair"],
            }
        )

    # if *really* nothing passed filters, still record that
    txt_path = img_out_dir / "inspection.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"Image: {image_path.name}\n")
        f.write(f"Annotated image: full.jpg\n")
        f.write(f"Total defects explained by LLaVA: {len(defect_summaries)}\n\n")

        if not defect_summaries:
            f.write(
                "No defects were sent to LLaVA. "
                "Either YOLO detected none, or all were too tiny/low-confidence.\n"
            )
        else:
            for d in defect_summaries:
                f.write(f"[Defect {d['id']}] conf={d['confidence']:.2f}\n")
                f.write(f"  defect: {d['defect']}\n")
                f.write(f"  cause:  {d['cause']}\n")
                f.write(f"  repair: {d['repair']}\n\n")

    print(f"✅ Output folder: {img_out_dir}")
    return img_out_dir


def inspect_folder(folder: Path):
    """
    Loop over all images in a folder and run the inspector.
    """
    print(f"\n📂 Scanning folder: {folder}")
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = [p for p in folder.iterdir() if p.suffix.lower() in exts]

    if not image_paths:
        print("No images found.")
        return

    print(f"Found {len(image_paths)} images.")

    # Load YOLO model ONCE
    print(f"📦 Loading YOLO model from: {YOLO_MODEL_PATH}")
    model = YOLO(str(YOLO_MODEL_PATH))

    for img_path in image_paths:
        try:
            inspect_image(model, img_path)
        except Exception as e:
            print(f"⚠️ Error on {img_path.name}: {e}")


# ===================== ENTRYPOINT =====================

if __name__ == "__main__":
    inspect_folder(IMAGES_FOLDER)
