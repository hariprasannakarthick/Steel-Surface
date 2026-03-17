from ultralytics import YOLO
import pyrealsense2 as rs
import cv2
import numpy as np  # ✅ You need this!
# Load your trained YOLO model
model = YOLO(r"C:\Users\HARI JHON\Downloads\Steel Surface\runs\detect\steel_defects_v1\weights\best.pt")

# Initialize RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

print("🎥 Camera started... Press 'ESC' to quit")

try:
    while True:
        # Get frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert to numpy array
        img = np.asanyarray(color_frame.get_data())

        # Run YOLO inference on current frame
        results = model.predict(source=img, show=True, conf=0.4, verbose=False)

        # Exit on 'ESC'
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    # Stop camera pipeline safely
    pipeline.stop()
    cv2.destroyAllWindows()
    print("🛑 Camera stopped.")

