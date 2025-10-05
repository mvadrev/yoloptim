import cv2
import time
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

prev_time = time.time()
fps_smooth = 0.0
alpha = 0.9  # smoothing factor for FPS
frame_count = 0
last_annotated = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % 3 == 0:
        results = model(frame, imgsz=416, conf=0.5)
        annotated_frame = results[0].plot()
        last_annotated = annotated_frame
    else:
        if last_annotated is not None:
            annotated_frame = last_annotated.copy()
        else:
            annotated_frame = frame.copy()

    curr_time = time.time()
    instant_fps = 1 / (curr_time - prev_time)
    fps_smooth = alpha * fps_smooth + (1 - alpha) * instant_fps
    prev_time = curr_time

    display_frame = annotated_frame.copy()
    cv2.putText(display_frame, f"FPS: {fps_smooth:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLO Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
