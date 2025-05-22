from ultralytics import YOLO

try:
    model = YOLO("yolov10x.pt")
    print("Model loaded successfully")
except Exception as e:
    print("Error loading model:", e)
