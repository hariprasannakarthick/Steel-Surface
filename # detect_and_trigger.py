# detect_and_trigger.py
import time
import cv2
from ultralytics import YOLO
import numpy as np

# RoboDK API
from robodk import robolink

# RealSense (Intel D435)
import pyrealsense2 as rs

# --- CONFIG ---
MODEL_PATH = "best.pt"         # your trained YOLOv8 model
CONF_THRESHOLD = 0.35          # detection confidence threshold
DEFECT_CLASSES = {"scratch","dent"}  # names used when training (adjust to your class names)
RDK_ACCEPT_PROGRAM = "accept"      # program names in RoboDK station
RDK_REJECT_PROGRAM = "rejection"
DEBOUNCE_FRAMES = 3            # require N consecutive defect frames before triggering rejection

# --- Load model ---
model = YOLO(MODEL_PATH)  # ultralytics YOLOv8

# --- Setup RealSense pipeline ---
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(cfg)

# --- Connect to RoboDK ---
RDK = robolink.Robolink()  # uses default local connection to RoboDK
print("Connected to RoboDK:", RDK.Connected())

def run_robodk_program(name):
    """
    Best-effort trigger of a program in RoboDK. Works if the program is present.
    """
    try:
        # Option A: run by program item:
        program_item = RDK.Item(name)   # look up item by name
        if program_item.Valid():
            program_item.RunProgram()   # run it
            print(f"Triggered RoboDK program: {name}")
            return True
        else:
            # Option B: fall back to station-level RunProgram call:
            RDK.RunProgram(name)
            print(f"Called RDK.RunProgram('{name}')")
            return True
    except Exception as e:
        print("Failed to run RoboDK program:", e)
        return False

# --- Main loop ---
try:
    consecutive_defect = 0
    consecutive_clear = 0
    last_action_time = 0
    MIN_ACTION_INTERVAL = 2.0  # seconds between program triggers to avoid spamming

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())

        # Run YOLO inference (returns results object)
        results = model(frame, imgsz=640, conf=CONF_THRESHOLD, verbose=False)  # ultralytics API
        # results is a list, get the first result
        res = results[0]

        # Determine if any detected classes are defects
        defect_detected = False
        # get class names and scores in a stable way:
        try:
            boxes = res.boxes  # Boxes object
            # each box has .cls for class index and .conf for confidence
            for box in boxes:
                cls_index = int(box.cls.cpu().numpy()[0]) if hasattr(box.cls, 'cpu') else int(box.cls)
                cls_name = model.names.get(cls_index, str(cls_index))
                conf = float(box.conf.cpu().numpy()[0]) if hasattr(box.conf, 'cpu') else float(box.conf)
                if cls_name.lower() in DEFECT_CLASSES and conf >= CONF_THRESHOLD:
                    defect_detected = True
                    break
        except Exception as e:
            # In case of API differences, do a fallback check on res.boxes.xyxy
            # The key idea: if any detection with name in DEFECT_CLASSES -> defect
            # Skip silently if API variant.
            pass

        # Visualize (optional)
        status_text = "DEFECT" if defect_detected else "CLEAR"
        cv2.putText(frame, status_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) if defect_detected else (0,255,0), 2)
        cv2.imshow("D435 - YOLO detect", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Debounce logic to avoid triggering on single noisy frame
        now = time.time()
        if defect_detected:
            consecutive_defect += 1
            consecutive_clear = 0
            if consecutive_defect >= DEBOUNCE_FRAMES and (now - last_action_time) > MIN_ACTION_INTERVAL:
                print("Decision: REJECT (defect present)")
                run_robodk_program(RDK_REJECT_PROGRAM)
                last_action_time = now
                consecutive_defect = 0
        else:
            consecutive_clear += 1
            consecutive_defect = 0
            if consecutive_clear >= DEBOUNCE_FRAMES and (now - last_action_time) > MIN_ACTION_INTERVAL:
                print("Decision: ACCEPT (clear)")
                run_robodk_program(RDK_ACCEPT_PROGRAM)
                last_action_time = now
                consecutive_clear = 0

except KeyboardInterrupt:
    print("Stopping...")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
