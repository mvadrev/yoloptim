import cv2
import numpy as np
import onnxruntime as ort
import time

# Load ONNX model
session = ort.InferenceSession("yolov8n.onnx", providers=["CPUExecutionProvider"])

# Get input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Set input size (depends on model export)
input_size = 640  # Or 416 if you exported with that

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

prev_time = 0

def preprocess(frame):
    img = cv2.resize(frame, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dim
    return img

while True:
    ret, frame = cap.read()
    if not ret:
        break

    curr_time = time.time()

    img_input = preprocess(frame)

    # Run inference
    outputs = session.run([output_name], {input_name: img_input})[0]

    # Post-process (VERY simplified, not production-ready)
    detections = outputs[0]  # shape: (batch=1, 84, 8400)

    # Draw bounding boxes (mockup - use real NMS + decode in production)
    for det in detections.T:
        conf = det[4]
        if conf > 0.5:
            class_id = int(np.argmax(det[5:]))
            x_center, y_center, width, height = det[0:4]
            x1 = int((x_center - width / 2) * frame.shape[1] / input_size)
            y1 = int((y_center - height / 2) * frame.shape[0] / input_size)
            x2 = int((x_center + width / 2) * frame.shape[1] / input_size)
            y2 = int((y_center + height / 2) * frame.shape[0] / input_size)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Class {class_id} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show FPS
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 ONNX Inference", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
