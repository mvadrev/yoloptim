import cv2
import time
from ultralytics import YOLO
import threading
import queue

model = YOLO('yolov8n.pt')

frame_queue = queue.Queue(maxsize=5)
result_queue = queue.Queue(maxsize=5)
stop_flag = threading.Event()

def capture_thread():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        stop_flag.set()
        return

    while not stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            stop_flag.set()
            break
        try:
            frame_queue.put(frame, timeout=1)
        except queue.Full:
            # Drop frame if full
            pass
    cap.release()

def inference_thread():
    frame_count = 0
    last_annotated = None
    while not stop_flag.is_set():
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue
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

        try:
            result_queue.put(annotated_frame, timeout=1)
        except queue.Full:
            pass

        frame_queue.task_done()

def display_thread():
    prev_time = time.time()
    fps_smooth = 0.0
    alpha = 0.9

    while not stop_flag.is_set():
        try:
            annotated_frame = result_queue.get(timeout=1)
        except queue.Empty:
            continue

        curr_time = time.time()
        instant_fps = 1 / (curr_time - prev_time)
        fps_smooth = alpha * fps_smooth + (1 - alpha) * instant_fps
        prev_time = curr_time

        display_frame = annotated_frame.copy()
        cv2.putText(display_frame, f"FPS: {fps_smooth:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("YOLO Detection", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag.set()

        result_queue.task_done()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    threads = [
        threading.Thread(target=capture_thread, daemon=True),
        threading.Thread(target=inference_thread, daemon=True),
        threading.Thread(target=display_thread, daemon=True),
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join()
