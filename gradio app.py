from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from ultralytics import YOLO
import ollama
import gradio as gr

# ===================== CONFIG =====================

# Base project folder
BASE_DIR = Path(r"D:\submersible pump impeller defect Detection analysis")

# Path to your finetuned YOLO model
YOLO_MODEL_PATH = Path(
    r"D:\submersible pump impeller defect Detection analysis\Finetuned Model.pt"
)

# Ollama model name -> check `ollama list`
LLAVA_MODEL_NAME = "llava:7b"   # or "llava" if that's what you see

# Root output folder (for logs / crops)
OUTPUT_DIR = BASE_DIR / "inspector_outputs_ui"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# YOLO detection params
CONF_THRES = 0.20      # lower = more recall
IOU_THRES = 0.5
IMG_SIZE = 512

# High recall but skip garbage:
MIN_LLAVA_CONF = 0.10      # send almost all reasonably-confident boxes
MIN_BOX_SIZE_FRAC = 0.002  # only skip tiny specks (<0.2% of image area)

# ===================== DEVICE & MODEL =====================

# 👉 Force GPU usage; error if not available
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is NOT available in this environment. "
        "Install a CUDA build of PyTorch (e.g. torch==2.5.1+cu121) "
        "and ensure `python -c \"import torch; print(torch.cuda.is_available())\"` prints True."
    )

DEVICE = 0  # use GPU 0
print("✅ Using GPU (device=0)")

print(f"📦 Loading YOLO model from: {YOLO_MODEL_PATH}")
YOLO_MODEL = YOLO(str(YOLO_MODEL_PATH))


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


# ===================== CORE INSPECTION (for UI) =====================

def inspect_image_ui(model: YOLO, image_path: Path) -> Tuple[Image.Image, str]:
    """
    Run YOLO + LLaVA on a single image path.
    Returns:
      - annotated full PIL image with boxes
      - short text summary of defects
    Also logs files into OUTPUT_DIR/image_stem/ for debugging.
    """
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    print(f"\n🖼️ Inspecting: {image_path.name}")

    results = model.predict(
        source=str(image_path),
        conf=CONF_THRES,
        iou=IOU_THRES,
        imgsz=IMG_SIZE,
        device=DEVICE,
        save=False,
        verbose=False,
    )[0]

    h, w = results.orig_shape
    orig_img = Image.open(image_path).convert("RGB")

    # Folder for this image (for logs / crops)
    img_out_dir = OUTPUT_DIR / image_path.stem
    crops_dir = img_out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Annotated full image from YOLO
    annotated_np = results.plot()                      # BGR np array
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

    # Build a short, human-readable summary string for UI
    lines = []
    if not defect_summaries:
        lines.append("No defects were confidently detected in this image.")
    else:
        for d in defect_summaries:
            lines.append(f"### Defect {d['id']} (conf={d['confidence']:.2f})")
            lines.append(f"- **Defect:** {d['defect'] or 'N/A'}")
            lines.append(f"- **Cause:** {d['cause'] or 'N/A'}")
            lines.append(f"- **Repair:** {d['repair'] or 'N/A'}")
            lines.append("")  # blank line

    summary_text = "\n".join(lines).strip()
    return annotated_img, summary_text


# ===================== GRADIO UI WRAPPERS =====================

def run_single_image(image: Image.Image):
    """Gradio wrapper for single-image mode."""
    if image is None:
        return None, "⚠ Please upload an image."

    temp_path = OUTPUT_DIR / "ui_single_input.jpg"
    image.save(temp_path)

    annotated, summary = inspect_image_ui(YOLO_MODEL, temp_path)
    return annotated, summary


def run_multiple_images(files: List[str]):
    """
    Gradio wrapper for multi-image mode.
    `files` will be a list of filepaths (because type='filepath').
    """
    if not files:
        return [], "⚠ Please upload one or more images."

    gallery_images = []
    all_text_blocks = []

    for file_path in files:
        p = Path(file_path)
        try:
            annotated, summary = inspect_image_ui(YOLO_MODEL, p)
            gallery_images.append(annotated)

            title = f"## {p.name}"
            all_text_blocks.append(title)
            if summary:
                all_text_blocks.append(summary)
            all_text_blocks.append("")  # blank line
        except Exception as e:
            all_text_blocks.append(f"### {p.name}\n- Error: {e}")
            all_text_blocks.append("")

    combined_text = "\n".join(all_text_blocks).strip()
    return gallery_images, combined_text or "No results."


def toggle_mode(mode):
    """Show/hide correct input component based on mode."""
    if mode == "Single image":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)


# ===================== THEME & APP =====================

theme = gr.themes.Soft(
    primary_hue="orange",
    neutral_hue="gray",
).set(
    body_background_fill="white",
    body_text_color="#222222",
    block_title_text_color="#ff7a1a",
    button_primary_background_fill="#ff7a1a",
    button_primary_background_fill_hover="#ff8f3c",
    button_primary_text_color="white",
)

custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}
#title-bar {
    background: linear-gradient(90deg, #ff7a1a, #ffb347);
    color: white;
    padding: 14px 18px;
    border-radius: 12px;
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 10px;
}
"""

with gr.Blocks(
    title="Submersible pump impeller defect Detection & Analysis",
    theme=theme,
    css=custom_css,
) as demo:
    gr.HTML(
        """
        <div id="title-bar">
            Submersible pump impeller defect Detection &amp; Analysis
        </div>
        """
    )

    gr.Markdown(
        """
Use this tool to automatically **detect submersible pump impeller defects**
and get a **short explanation** for each:

- Left: choose **Single image** or **Multiple images**, then drag & drop.
- For multiple: drag-select MANY images from Explorer and drop them here (browsers can't drop raw folders).
- Right (top): annotated image(s) with defect bounding boxes.
- Right (bottom): concise explanation (**defect, cause, repair**).
"""
    )

    with gr.Row():
        # LEFT SIDE: Controls
        with gr.Column(scale=1):
            mode = gr.Radio(
                ["Single image", "Multiple images"],
                value="Single image",
                label="Mode",
            )

            single_input = gr.Image(
                label="Upload / drag-drop single image",
                type="pil",
                visible=True,
            )

            multi_input = gr.Files(
                label="Upload / drag-drop multiple images",
                file_types=["image"],
                file_count="multiple",
                type="filepath",
                visible=False,
            )

            run_button = gr.Button("Run Inspection", variant="primary")

        # RIGHT SIDE: Outputs
        with gr.Column(scale=2):
            output_gallery = gr.Gallery(
                label="Annotated defect images",
                show_label=True,
                columns=2,
                height=400,
            )

            output_text = gr.Markdown(
                label="Defect explanations",
            )

    mode.change(
        toggle_mode,
        inputs=mode,
        outputs=[single_input, multi_input],
    )

    def on_run(mode_value, img, files):
        if mode_value == "Single image":
            annotated, summary = run_single_image(img)
            return [annotated] if annotated is not None else [], summary
        else:
            gallery, text = run_multiple_images(files)
            return gallery, text

    run_button.click(
        on_run,
        inputs=[mode, single_input, multi_input],
        outputs=[output_gallery, output_text],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
