# Real-time-object-detection

## 🧠 **Project Summary**

This script uses the **YOLOv8 object detection model** from the **Ultralytics** library to perform real-time detection on webcam input. Simultaneously, it shows a **live Matplotlib plot** tracking how many objects were detected over time — both in **separate windows**.

---

## 🔍 **Code Description (Section-wise)**

---

### 📦 **Imports**

```python
import cv2, numpy as np
from ultralytics import YOLO
from collections import deque
import matplotlib.pyplot as plt
import threading
import time
```

* `cv2`: Access webcam and show video feed.
* `YOLO`: Load YOLOv8 model for object detection.
* `deque`: Efficient history storage for tracking.
* `matplotlib.pyplot`: Plot object counts.
* `threading`: Run plotting in a background thread.
* `time`: Track timestamps and delays.

---

### 🤖 **Model Loading**

```python
model = YOLO("yolov8n.pt")
```

Loads the YOLOv8 *nano* version (lightweight and fast) pre-trained on the COCO dataset.

---

### 🧠 **Tracking Initialization**

```python
track_history = {}
max_history = 10
```

Keeps short-term movement history (10 frames) for each detected object to determine its direction.

---

### 🎥 **Webcam Setup**

```python
cap = cv2.VideoCapture(0)
```

Opens the default webcam for video input.

---

### 📊 **Data Storage**

```python
object_counts = deque(maxlen=100)
timestamps = deque(maxlen=100)
```

Stores object count and time data for the last 100 frames for plotting.

---

### 📈 **Live Plot Thread**

```python
def live_plot(): ...
```

A background function:

* Uses `plt.ion()` for live plotting.
* Updates every 0.3 seconds.
* Displays object count over time in a separate chart window.

This is run in a thread using:

```python
plot_thread = threading.Thread(target=live_plot)
plot_thread.start()
```

---

### ➡️ **Direction Helper**

```python
def get_direction(pts): ...
```

Calculates the movement direction of a tracked object based on its recent positions:

* Horizontal (`Right` / `Left`)
* Vertical (`Up` / `Down`)

---

### 🔄 **Main Loop**

```python
while True:
    ...
    results = model(frame, stream=True)
```

* Captures webcam frame.
* Passes it to YOLOv8 for detection (stream mode allows efficient iteration).

---

### 📦 **For Each Detection**

```python
for box in boxes:
    ...
```

* Extracts object info (label, confidence, bounding box).
* Draws rectangle & label on the frame.
* Tracks object center and stores it in `track_history`.
* Displays direction and object name on screen.
* Counts all detected objects.

---

### ⏱️ **Update Timestamp and Count**

```python
current_time = round(time.time() - start_time, 1)
timestamps.append(current_time)
object_counts.append(current_count)
```

Updates the plot data for each frame with time and count.

---

### 🖼️ **Show Frame**

```python
cv2.imshow("YOLOv8 Live Object Detection", frame)
```

Displays the detection output in OpenCV's window.

---

### ❌ **Quit and Cleanup**

```python
if cv2.waitKey(1) == ord('q'):
    break
```

* Press 'q' to stop.
* Gracefully stops the webcam and closes windows.
* Stops the plot thread.

---

## ✅ **Output**

You will see:

1. **YOLOv8 Live Object Detection** window showing detections with labels, confidence, and direction.
2. A **Matplotlib plot window** showing how object counts have changed over time.


