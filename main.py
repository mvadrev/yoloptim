import cv2
import time
from ultralytics import YOLO

# Load YOLO model (use 'yolov11n.pt' or custom model if needed)
model = YOLO('yolov8n.pt')  # or 'yolov11n.pt'

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

# For FPS calculation
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Start time
    curr_time = time.time()

    # Run YOLO inference
    results = model(frame, imgsz=416, conf=0.5)

    # Annotate frame with bounding boxes
    annotated_frame = results[0].plot()

    # Calculate FPS
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # Draw FPS on the frame
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display
    cv2.imshow("YOLO Detection", annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
