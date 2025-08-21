import os
import matplotlib as mpl
from ultralytics import YOLO

# 모델 로드
model = YOLO("runs/detect/train/weights/last.pt")

# 학습
model.train(
    data="data.yaml",
    imgsz=640,          
    epochs=50,
    batch=0,            
    device=0,
    workers=8,
    amp=True,
    patience=10,
    close_mosaic=5,
    plots=True,       
)