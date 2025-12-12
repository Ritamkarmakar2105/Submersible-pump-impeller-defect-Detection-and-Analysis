from ultralytics import YOLO
import torch
import os

# ======= PATHS (your exact ones) =======
BASE_DIR = r"D:\submersible pump impeller defect Detection analysis"

MODEL_PATH = r"D:\submersible pump impeller defect Detection analysis\yolo11l.pt"
DATA_YAML = r"D:\submersible pump impeller defect Detection analysis\Annotated dataset\data.yaml"

PROJECT_DIR = os.path.join(BASE_DIR, "training_runs")
RUN_NAME = "yolo11l_casting_annotated_714"
# =======================================

# Training hyperparameters tuned for higher accuracy
EPOCHS = 200        # long training for better convergence
IMG_SIZE = 512
BATCH_SIZE = 16     # reduce to 8 or 4 if you get CUDA OOM


def main():
    print("DATA_YAML ->", DATA_YAML)
    print("MODEL_PATH ->", MODEL_PATH)

    cuda_ok = torch.cuda.is_available()
    print("CUDA available:", cuda_ok)

    if cuda_ok:
        device = 0
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        print("⚠ CUDA not available, training will run on CPU (slow).")

    # 1) Load YOLO11-L base weights
    model = YOLO(MODEL_PATH)

    # 2) Train for high accuracy
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=device,
        workers=4,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,

        # “industrial-ish” training tweaks
        cos_lr=True,          # cosine learning-rate schedule
        patience=80,          # allow long training before early stop
        augment=True,         # enable default augmentations
        lr0=0.01,             # initial LR (default, but explicit)
        lrf=0.01,             # final LR fraction
        weight_decay=0.0005,  # regularization
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        mosaic=1.0,
        mixup=0.0,
    )

    print("\n🎯 Training complete on 714 annotated images with YOLO11-L!")
    print("Results directory:", results.save_dir)
    print("Best model will be at:")
    print(os.path.join(results.save_dir, "weights", "best.pt"))


if __name__ == "__main__":
    main()
