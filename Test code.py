from ultralytics import YOLO

# Load your trained model
model = YOLO(r"C:\Users\HARI JHON\Downloads\Steel Surface\runs\detect\steel_defects_v1\weights\best.pt")

# Run inference (test) on an image or folder of images
results = model.predict(
    source=r"C:\Users\HARI JHON\Downloads\Steel Surface\WhatsApp Image 2025-11-04 at 10.47.35_936bb581.jpg",  # change to your test image path
    show=True,          # shows image window with detections
    save=True,          # saves results automatically
    conf=0.5            # confidence threshold (adjust if needed)
)

print("✅ Inference complete! Check the 'runs/predict' folder for results.")
