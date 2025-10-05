import cv2
import numpy as np
import onnxruntime as ort
import time

# Load ONNX model
session = ort.InferenceSession("yolov8n_fp16.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

input_size = 640

def preprocess(frame):
    img = cv2.resize(frame, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = img.astype(np.float16) / 255.0  # float16 input
    img = img.astype(np.float16) / 255.0  # float16 input

    img = np.transpose(img, (2, 0, 1))    # HWC to CHW
    img = np.expand_dims(img, axis=0)     # Add batch
    return img

def xywh_to_xyxy(x, y, w, h):
    return x - w / 2, y - h / 2, x + w / 2, y + h / 2

def non_max_suppression(boxes, scores, conf_threshold=0.3, iou_threshold=0.45):
    if not boxes:
        return []

    boxes_int = [[
        int(b[0] * input_size), int(b[1] * input_size),
        int((b[2] - b[0]) * input_size), int((b[3] - b[1]) * input_size)
    ] for b in boxes]

    indices = cv2.dnn.NMSBoxes(boxes_int, scores, conf_threshold, iou_threshold)
    if len(indices) == 0:
        return []
    return indices.flatten()

def postprocess(output, conf_threshold=0.3):
    pred = output.squeeze().T  # shape (8400, 84)
    boxes = []
    scores = []

    for det in pred:
        conf_obj = det[4]
        if conf_obj < conf_threshold or not np.isfinite(conf_obj):
            continue

        class_scores = det[5:]
        class_id = np.argmax(class_scores)
        class_conf = class_scores[class_id]
        conf = conf_obj * class_conf

        if conf < conf_threshold or not np.isfinite(conf):
            continue

        x_c, y_c, w, h = det[0:4]
        x1, y1, x2, y2 = xywh_to_xyxy(x_c, y_c, w, h)

        # Clamp coordinates between 0 and 1
        x1, y1, x2, y2 = [max(0.0, min(1.0, v)) for v in (x1, y1, x2, y2)]

        boxes.append([x1, y1, x2, y2])
        scores.append(float(conf))

    keep = non_max_suppression(boxes, scores, conf_threshold, 0.45)

    if len(keep) == 0:
        return [], []

    return [boxes[i] for i in keep], [scores[i] for i in keep]

cap = cv2.VideoCapture(0)
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = preprocess(frame)
    outputs = session.run([output_name], {input_name: input_tensor})

    boxes, scores = postprocess(outputs[0], conf_threshold=0.3)

    for box, score in zip(boxes, scores):
        x1 = int(box[0] * frame.shape[1])
        y1 = int(box[1] * frame.shape[0])
        x2 = int(box[2] * frame.shape[1])
        y2 = int(box[3] * frame.shape[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("YOLOv8n FP16 ONNX Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
