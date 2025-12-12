from ultralytics import YOLO

MODEL_PATH = r"D:\submersible pump impeller defect Detection analysis\Finetuned Model.pt"
IMAGE_PATH = r"D:\submersible pump impeller defect Detection analysis\casting_data\casting_data\train\def_front\cast_def_0_1707.jpeg"  # <-- your image

model = YOLO(MODEL_PATH)

results = model.predict(
    source=IMAGE_PATH,
    conf=0.30,      # super low, show *anything* it sees
    iou=0.5,
    imgsz=512,
    device=0,
    save=True,
    show=True,
    verbose=True,
)

print("\nRaw detections (even very low confidence):")
for box in results[0].boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    xyxy = box.xyxy[0].tolist()
    print(f"class={cls_id}, conf={conf:.3f}, box={xyxy}")
