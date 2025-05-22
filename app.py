# from flask import Flask, render_template, Response, send_file
# import cv2
# import torch
# from ultralytics import YOLO
# from database import log_detection
# from report_generator import generate_report
# import clip
# import torch
# from PIL import Image
# import numpy as np
# from segment_anything import SamPredictor, sam_model_registry
# # from dinov2 import DINOv2
# # from twilio.rest import Client
# import logging
# import random

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)
# # try:
# #     from dinov2 import DINOv2
# # except ImportError as e:
# #     logging.error(f"Failed to import DINOv2: {e}")

# app = Flask(__name__)

# # Load YOLOv8 model
# model = YOLO("yolov8n.pt")  # Make sure yolov8n.pt is in the project folder
# cap = cv2.VideoCapture(0)  # Use webcam (or provide video file path)

# # Load CLIP model
# clip_model, preprocess = clip.load("ViT-B/32", device="cpu")

# # Load SAM model
# sam = sam_model_registry["vit_b"](checkpoint="sam_vit_h_4b8939.pth")
# sam_predictor = SamPredictor(sam)

# # # Load DINOv2 model
# # try:
# #     dino_model = DINOv2.load_model("dinov2.pth")
# # except NameError:
# #     dino_model = None
# #     logging.error("DINOv2 model is not available")

# def generate_frames():
#     object_colors = {}
#     person_count = 0
#     alert_classes = ["knife", "gun", "weapon"] 

#     while True:
#         success, frame = cap.read()
#         if not success:
#             logging.error("Failed to read frame from video capture")
#             break
        
#         # Perform object detection
#         results = model(frame)
#         person_count = 0
#         for r in results:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 conf = float(box.conf[0])
#                 cls = int(box.cls[0])
#                 label = f"{model.names[cls]} {conf:.2f}"

#                 # Log detections to the database
#                 log_detection(model.names[cls], conf)

#                 # Count the number of persons
#                 if model.names[cls] == "person":
#                     person_count += 1

#                 # Assign a unique color to each object class
#                 if cls not in object_colors:
#                     object_colors[cls] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
#                 color = object_colors[cls]

#                 # Turn the box red if a weapon is detected
#                 if model.names[cls] in alert_classes:
#                     color = (0, 0, 255)
#                     cv2.putText(frame, "ALERT!", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#                 # Draw bounding box
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#         # Display person count on the frame
#         cv2.putText(frame, f"Persons: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

#         logging.debug("Object detection completed")
#         # Process image with CLIP
#         image = preprocess(Image.fromarray(frame)).unsqueeze(0)
#         with torch.no_grad():
#             clip_features = clip_model.encode_image(image)
#         logging.debug("CLIP processing completed")
#         cv2.putText(frame, "CLIP Processed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
#         # Process image with SAM
#         sam_predictor.set_image(frame)
#         masks, _, _ = sam_predictor.predict()
#         logging.debug("SAM processing completed")
#         cv2.putText(frame, "SAM Processed", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
#     #    # Process image with DINOv2
#     #     if dino_model:
#     #         dino_features = dino_model.extract_features(np.array(frame))
#     #         logging.debug("DINOv2 processing completed")
#     #         cv2.putText(frame, "DINOv2 Processed", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         _, buffer = cv2.imencode('.jpg', frame)
#         frame_bytes = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/generate_report')
# def generate_report_route():
#     report_file = generate_report()
#     if report_file:
#         send_whatsapp_alert(report_file)
#         return send_file(report_file, as_attachment=True)
#     else:
#         return "No recent detections to generate a report.", 400

# if __name__ == '__main__':
#     app.run(debug=True)


# // 

# import cv2
# import threading
# import queue
# import time
# import logging
# from flask import Flask, render_template, Response
# from ultralytics import YOLO
# from segment_anything import sam_model_registry, SamPredictor
# from openvino.runtime import Core

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)

# # Initialize Flask
# app = Flask(__name__)

# # Load YOLOv11m model with OpenVINO
# core = Core()
# model = YOLO("yolov11m_openvino_model.xml")  # OpenVINO model
# model.to("AUTO")  # Auto-selects best Intel device

# # Load SAM model
# sam_checkpoint = "sam_vit_h_4b8939.pth"
# sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to("cpu")  # No CUDA
# sam_predictor = SamPredictor(sam)

# # OpenCV Video Capture
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cap.set(cv2.CAP_PROP_FPS, 30)

# # Alert Classes
# alert_classes = ["knife", "gun", "weapon"]
# frame_queue = queue.Queue(maxsize=5)  # Limit queue size for real-time processing

# # Background thread to read frames
# def capture_frames():
#     while True:
#         success, frame = cap.read()
#         if success:
#             if not frame_queue.full():
#                 frame_queue.put(frame)
#         time.sleep(0.01)  # Prevent overloading

# # Thread for processing frames
# def process_frames():
#     global processed_frame
#     while True:
#         if not frame_queue.empty():
#             frame = frame_queue.get()

#             # YOLO Object Detection
#             results = model(frame)
#             for r in results:
#                 for box in r.boxes:
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     conf = float(box.conf[0])
#                     cls = int(box.cls[0])
#                     label = f"{model.names[cls]} {conf:.2f}"

#                     # If weapon detected, use red color
#                     color = (0, 255, 0)  # Default green
#                     if model.names[cls] in alert_classes:
#                         color = (0, 0, 255)
#                         cv2.putText(frame, "ALERT!", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#                     # Draw bounding box
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                     cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#             processed_frame = frame  # Store processed frame

# # Start background threads
# threading.Thread(target=capture_frames, daemon=True).start()
# threading.Thread(target=process_frames, daemon=True).start()

# # Flask Route for Video Streaming
# def generate_frames():
#     while True:
#         if "processed_frame" in globals():
#             _, buffer = cv2.imencode('.jpg', processed_frame)
#             frame_bytes = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(debug=True, threaded=True, use_reloader=False)


# from flask import Flask, render_template, Response, send_file
# import cv2
# import torch
# import threading
# from ultralytics import YOLO
# from database import log_detection
# from report_generator import generate_report
# import clip
# from PIL import Image
# import numpy as np
# from segment_anything import SamPredictor, sam_model_registry
# import logging
# import random

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)

# app = Flask(__name__)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# logging.info(f"Using device: {device}")

# # Load models on GPU
# model = YOLO("yolov8n.pt").to(device)
# cap = cv2.VideoCapture(0)
# clip_model, preprocess = clip.load("ViT-B/32", device=device)

# sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").to(device)
# sam_predictor = SamPredictor(sam)

# # Multi-threaded frame processing
# frame_lock = threading.Lock()
# latest_frame = None


# def capture_frames():
#     global latest_frame
#     while True:
#         success, frame = cap.read()
#         if success:
#             with frame_lock:
#                 latest_frame = frame.copy()
#         else:
#             logging.error("Failed to read frame from video capture")


# def process_frame():
#     global latest_frame
#     object_colors = {}
#     alert_classes = ["knife", "gun", "weapon"]
    
#     while True:
#         with frame_lock:
#             if latest_frame is None:
#                 continue
#             frame = latest_frame.copy()
        
#         frame_tensor = torch.from_numpy(frame).to(device).float()
#         results = model(frame_tensor)
#         person_count = 0
        
#         for r in results:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 conf = float(box.conf[0])
#                 cls = int(box.cls[0])
#                 label = f"{model.names[cls]} {conf:.2f}"
#                 log_detection(model.names[cls], conf)
                
#                 if model.names[cls] == "person":
#                     person_count += 1
                
#                 if cls not in object_colors:
#                     object_colors[cls] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
#                 color = object_colors[cls]
                
#                 if model.names[cls] in alert_classes:
#                     color = (0, 0, 255)
#                     cv2.putText(frame, "ALERT!", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
#         cv2.putText(frame, f"Persons: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
#         image = preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
#         with torch.no_grad():
#             clip_model.encode_image(image)
        
#         sam_predictor.set_image(frame)
#         sam_predictor.predict()
        
#         _, buffer = cv2.imencode('.jpg', frame)
#         yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(process_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/generate_report')
# def generate_report_route():
#     report_file = generate_report()
#     if report_file:
#         return send_file(report_file, as_attachment=True)
#     return "No recent detections to generate a report.", 400

# if __name__ == '__main__':
#     threading.Thread(target=capture_frames, daemon=True).start()
#     app.run(debug=True, threaded=True)

from flask import Flask, render_template, Response, send_file
import cv2
import torch
from ultralytics import YOLO
from database import log_detection
from report_generator import generate_report
import clip
from PIL import Image
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
# from dinov2 import DINOv2
import logging
import random
from twilio.rest import Client
from dotenv import load_dotenv
import os

load_dotenv()


account_sid = os.getenv('account_sid')
auth_token = os.getenv('auth_token')
sent_from = os.getenv('sent_from')
sent_to = os.getenv('sent_to')

logging.basicConfig(level=logging.DEBUG)
# try:
#     from dinov2 import DINOv2
# except ImportError as e:
#     logging.error(f"Failed to import DINOv2: {e}")

app = Flask(__name__)

model = YOLO("yolo11m.pt")  # Make sure yolov8n.pt is in the project folder
cap = cv2.VideoCapture(0)  # Use webcam (or provide video file path)

# clip_model, preprocess = clip.load("ViT-B/32", device="cpu")


# sam = sam_model_registry["vit_b"](checkpoint="sam_vit_h_4b8939.pth")
# sam_predictor = SamPredictor(sam)

# try:
#     dino_model = DINOv2.load_model("dinov2.pth")
# except NameError:
#     dino_model = None
#     logging.error("DINOv2 model is not available")

def generate_frames():
    object_colors = {}
    person_count = 0
    alert_classes = ["knife", "gun", "weapon","stick"] #but sirf whi classes that are defined in the yolov11 mei 

    while True:
        success, frame = cap.read()
        if not success:
            logging.error("Failed to read frame from video capture")
            break
        
        # Perform object detection
        results = model(frame)
        person_count = 0
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{model.names[cls]} {conf:.2f}"

                # if(label==alert_classes[3]):
                #     print(label)

                # Log detections to the database
                log_detection(model.names[cls], conf)

                # Count the number of persons
                if model.names[cls] == "person":
                    person_count += 1

                # Assign a unique color to each object class
                if cls not in object_colors:
                    object_colors[cls] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                color = object_colors[cls]

                # Turn the box red if a weapon is detected
                if model.names[cls] in alert_classes:
                    color = (0, 0, 255)
                    cv2.putText(frame, "ALERT!", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    client = Client(account_sid, auth_token)
                    message = client.messages.create(
                    from_=sent_from,
                    body=f"{model.names[cls]} is spotted vist to reduce the chance of any attack",
                    to=sent_to
                    )
                    print(message.sid)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display person count on the frame
        cv2.putText(frame, f"Persons: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        logging.debug("Object detection completed")
        # Process image with CLIP
        # image = preprocess(Image.fromarray(frame)).unsqueeze(0)
        # with torch.no_grad():
        #     clip_features = clip_model.encode_image(image)
        # logging.debug("CLIP processing completed")
        # cv2.putText(frame, "CLIP Processed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Process image with SAM
        # sam_predictor.set_image(frame)
        # masks, _, _ = sam_predictor.predict()
        # logging.debug("SAM processing completed")
        # cv2.putText(frame, "SAM Processed", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
    #    # Process image with DINOv2
    #     if dino_model:
    #         dino_features = dino_model.extract_features(np.array(frame))
    #         logging.debug("DINOv2 processing completed")
    #         cv2.putText(frame, "DINOv2 Processed", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_report')
def generate_report_route():
    
    report_file = generate_report()
    if report_file:
        # send_whatsapp_alert(report_file)
        return send_file(report_file, as_attachment=True)
    else:
        return "No recent detections to generate a report.", 400

if __name__ == '__main__':
    app.run(debug=True)