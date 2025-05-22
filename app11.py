import cv2
import time
import random
import os
import logging
import numpy as np
from flask import Flask, render_template, Response, send_file
from flask_cors import CORS
from ultralytics import YOLO
from dotenv import load_dotenv
from database import log_detection
from report_generator import generate_report
from twilio.rest import Client

load_dotenv()
# account_sid = os.getenv('account_sid')
# auth_token = os.getenv('auth_token')
# sent_from = os.getenv('sent_from')
# sent_to = os.getenv('sent_to')

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.ERROR)

model = YOLO("D:\\AI\\Cr\\yolo11m.pt")
cap = cv2.VideoCapture("D:\\AI\\Cr\\video\\fight.mp4")

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# For naive tracking using centroids
previous_centroids = {}

def generate_frames():
    frame_skip =4 
    frame_counter = 0
    allowed_classes = ["person", "knife", "gun", "crowbar"]
    alert_classes = ["knife", "gun", "crowbar"]

    global previous_centroids
    center_color_map = {}

    def get_color_for_center(center):
        key = (center[0] // 10, center[1] // 10)
        if key not in center_color_map:
            center_color_map[key] = (
                random.randint(100, 255),
                random.randint(100, 255),
                random.randint(100, 255)
            )
        return center_color_map[key]

    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            previous_centroids.clear()
            center_color_map.clear()
            continue

        frame_counter += 1
        if frame_counter % frame_skip != 0:
            continue

        frame = cv2.resize(frame, (640, 360))
        results = model(frame, verbose=False)
        detections = []
        person_centroids = []

        for r in results:
            for i, box in enumerate(r.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]

                if class_name not in allowed_classes:
                    continue

                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                detection = {
                    'box': (x1, y1, x2, y2),
                    'conf': conf,
                    'cls': cls,
                    'class_name': class_name,
                    'center': center,
                    'violence': False
                }

                if class_name == "person":
                    person_centroids.append(center)

                detections.append(detection)
                log_detection(class_name, conf)

            # Violence detection
            violence_pairs = set()
            for i in range(len(person_centroids)):
                for j in range(i + 1, len(person_centroids)):
                    dist = distance(person_centroids[i], person_centroids[j])
                    prev_i = previous_centroids.get(i)
                    prev_j = previous_centroids.get(j)
                    if dist < 50:
                        if prev_i and prev_j:
                            move_i = distance(person_centroids[i], prev_i)
                            move_j = distance(person_centroids[j], prev_j)
                            if move_i > 5 and move_j > 5:
                                violence_pairs.add((i, j))

            # Update centroids
            for idx, center in enumerate(person_centroids):
                previous_centroids[idx] = center

            # Draw boxes
            person_idx = 0
            for det in detections:
                x1, y1, x2, y2 = det['box']
                class_name = det['class_name']
                label = f"{class_name} {det['conf']:.2f}"

                color = get_color_for_center(det['center'])

                if class_name == "person":
                    for vi, vj in violence_pairs:
                        if person_idx == vi or person_idx == vj:
                            det['violence'] = True
                            label = "person(violence)"
                            color = (0, 0, 255)
                            cv2.putText(frame, "Violence Detected!", (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                           
                           
                            account_sid = 'AC0e2b70a7f5646447db78a700f070f3b9'
                            auth_token = '79fdad687eff3e3b4b4a942b0aaac3b7'
                            client = Client(account_sid, auth_token)
                            message = client.messages.create(
                            from_='+19896932477',
                            body=f"{model.names[cls]} is spotted vist to reduce the chance of any attack",
                            to='+918868070103'
                            )
                    person_idx += 1

                elif class_name in alert_classes:
                    
                    color = (0, 0, 255)
                    cv2.putText(frame, "ALERT!", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # account_sid = 'AC0e2b70a7f5646447db78a700f070f3b9'
                    # auth_token = '79fdad687eff3e3b4b4a942b0aaac3b7'
                    # client = Client(account_sid, auth_token)
                    # message = client.messages.create(
                    # from_='+19896932477',
                    # body=f"{model.names[cls]} is spotted vist to reduce the chance of any attack",
                    # to='+918868070103'
                    # )

                # Draw the box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.putText(frame, f"Persons: {person_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(0.03)


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
        return send_file(report_file, as_attachment=True)
    else:
        return "No recent detections to generate a report.", 400

if __name__ == '__main__':
    app.run(debug=True)
