from ultralytics import YOLO

# Load a pre-trained YOLO model
model = YOLO('yolov8n.pt')  # Example: YOLOv8 Nano

# Export the model to OpenVINO format
model.export(format='openvino')