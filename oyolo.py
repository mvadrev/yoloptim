import cv2
import time
from ultralytics import YOLO
import threading
import queue

# Load YOLO model
model = YOLO('yolov8n.pt')

# Queues for frames and results
frame_queue = queue.Queue(maxsize=10)
result_queue = queue.Queue(maxsize=10)

# Flag to signal threads to stop
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
            print("Error: Failed to grab frame")
            stop_flag.set()
            break
        try:
            frame_queue.put(frame, timeout=1)
        except queue.Full:
            # Drop frame if queue is full to keep real-time
            pass

    cap.release()

def inference_thread():
    while not stop_flag.is_set():
        try:
            while True:
                frame = frame_queue.get_nowait()
                frame_queue.task_done()
        except queue.Empty:
            pass

        if 'frame' in locals():
            results = model(frame, imgsz=416, conf=0.5)

            # Filter detections to only class 0 ("person")
            result = results[0]
            mask = result.boxes.cls == 0  # cls tensor, 0 is person in COCO
            filtered_boxes = result.boxes[mask]
            
            # Replace boxes in result with filtered ones
            result.boxes = filtered_boxes

            annotated_frame = result.plot()

            try:
                result_queue.put(annotated_frame, timeout=1)
            except queue.Full:
                pass
        else:
            time.sleep(0.01)



def display_thread():
    prev_time = 0
    while not stop_flag.is_set():
        try:
            annotated_frame = result_queue.get(timeout=1)
        except queue.Empty:
            continue

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("YOLO Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag.set()

        result_queue.task_done()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Create and start threads
    threads = [
        threading.Thread(target=capture_thread, daemon=True),
        threading.Thread(target=inference_thread, daemon=True),
        threading.Thread(target=display_thread, daemon=True),
    ]

    for t in threads:
        t.start()

    # Wait for all threads to finish
    for t in threads:
        t.join()
