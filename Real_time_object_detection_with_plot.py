import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import matplotlib.pyplot as plt
import threading
import time

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize tracking history
track_history = {}
max_history = 10

# Start webcam
cap = cv2.VideoCapture(0)

# Object count and timestamp history
object_counts = deque(maxlen=100)
timestamps = deque(maxlen=100)

start_time = time.time()
running = True

# Function to update live chart in a separate window
def live_plot():
    plt.ion()
    fig, ax = plt.subplots()
    while running:
        if timestamps:
            ax.clear()
            ax.plot(timestamps, object_counts, marker='o', color='blue')
            ax.set_title("Object Count Over Time")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Count")
            ax.set_ylim(0, max(5, max(object_counts)+1))
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.pause(0.1)
        time.sleep(0.3)

# Start the chart update in a background thread
plot_thread = threading.Thread(target=live_plot)
plot_thread.start()

# Direction calculation
def get_direction(pts):
    if len(pts) < 2:
        return ""
    dx = pts[-1][0] - pts[0][0]
    dy = pts[-1][1] - pts[0][1]
    if abs(dx) > abs(dy):
        return "Right" if dx > 0 else "Left"
    else:
        return "Down" if dy > 0 else "Up"

# Main video loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)
    current_count = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            obj_id = f"{label}{x1}{y1}"
            if obj_id not in track_history:
                track_history[obj_id] = deque(maxlen=max_history)
            track_history[obj_id].append(center)

            cv2.circle(frame, center, 5, (255, 0, 0), -1)
            direction = get_direction(track_history[obj_id])

            text = f"{label} {int(conf * 100)}% {direction}"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            current_count += 1

    current_time = round(time.time() - start_time, 1)
    timestamps.append(current_time)
    object_counts.append(current_count)

    cv2.imshow("YOLOv8 Live Object Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
running = False
plot_thread.join()
cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.close()