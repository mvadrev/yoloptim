from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.export(format='onnx', opset=12)  # exports as FP32 ONNX by default
