from ultralytics import YOLO
import torch
import os

BASE_DIR = r"D:\submersible pump impeller defect Detection analysis"

MODEL_PATH = os.path.join(BASE_DIR, "Trained model.pt")
DATA_YAML = os.path.join(
    BASE_DIR,
    r"Dataset for finetuning\data.yaml"
)

PROJECT_DIR = os.path.join(BASE_DIR, "training_runs")
RUN_NAME = "yolo11l_finetune_60_hardcases"

# Finetune hyperparameters
EPOCHS = 60        # shorter, focused training
IMG_SIZE = 512
BATCH_SIZE = 8     # safe for 6GB VRAM


def main():
    print("Using base model:", MODEL_PATH)
    print("Finetune data:", DATA_YAML)

    device = 0 if torch.cuda.is_available() else "cpu"
    print("CUDA available:", torch.cuda.is_available())
    if device == 0:
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU (slower).")

    # 1️⃣ Load your already-trained model
    model = YOLO(MODEL_PATH)

    # 2️⃣ Finetune on the 60 hard-case images
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

        # gentler training for finetuning
        cos_lr=True,
        patience=30,
        lr0=0.005,          # a bit lower LR than full training
        weight_decay=0.0005,
        augment=True,       # keep augmentations
    )

    print("\n✅ Finetune complete!")
    print("Results directory:", results.save_dir)
    print("New best model path:")
    print(os.path.join(results.save_dir, "weights", "best.pt"))


if __name__ == "__main__":
    main()
