# ============================================================
# STEEL SURFACE DEFECT DETECTION - YOLOv8 TRAINING SCRIPT
# Windows-safe version (with __main__ protection)
# ============================================================

from ultralytics import YOLO
import torch
import os

def main():
    # === DATASET PATH ===
    dataset_dir = r"C:\Users\HARI JHON\Downloads\Steel Surface\Steel Surface Defects.v1i.yolov8 (1)"
    data_yaml = os.path.join(dataset_dir, "data.yaml")

    # === DEVICE CHECK ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 Training on: {device.upper()}")
    if device == "cuda":
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ No GPU found — training will run on CPU (very slow)")

    # === MODEL SELECTION ===
    model = YOLO("yolov8s.pt")  # Change to yolov8n.pt or yolov8m.pt if needed

    # === TRAINING ===
    model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        project="runs/detect",
        name="steel_defects_v1",
        workers=0,              # 🔧 Set to 0 for Windows safety
        exist_ok=True,
        save=True,
        save_period=5,
        patience=15,
        pretrained=True,7
        optimizer="auto",
        verbose=True
    )

    # === VALIDATION ===
    metrics = model.val()
    print("\n📊 Validation metrics:", metrics)

    # === OPTIONAL TEST ===
    # test_img = r"C:\Users\HARI JHON\Downloads\Steel Surface\test.jpg"
    # results = model.predict(source=test_img, conf=0.5, show=True)
    # results.save(save_dir="runs/predict/steel_defects_samples")

if __name__ == "__main__":
    main()