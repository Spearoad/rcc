import torch
import torch.nn.functional as F
import timm
import gradio as gr
from PIL import Image
import torchvision.transforms as T
from ultralytics import YOLO
import numpy as np

device = torch.device("cpu")

# 분류 모델
num_classes = 3
classify_model = timm.create_model(
    "efficientnet_b3", pretrained=False, num_classes=num_classes)

# 학습된 가중치 불러오기
state_dict = torch.load("classify.pth", map_location="cpu")
classify_model.load_state_dict(state_dict)
classify_model.to(device)
classify_model.eval()

# 전처리
transform_cls = T.Compose([
    T.Resize((300, 300)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

def fine_to_coarse(idx: int) -> str:
    return "손상됨" if idx in (1, 2) else "정상"
    
# 탐지 모델
yolo_model = YOLO("detect.pt")   # 가중치 불러오기
yolo_model.to(device)
yolo_model.eval()

# 예측 함수
def pipeline_predict(img: Image.Image):
    # 분류
    img_cls = transform_cls(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out_cls = classify_model(img_cls)
        pred_cls = torch.argmax(out_cls, dim=1).item()

    # 손상됨 상태면 탐지
    if pred_cls == 0:
        return "정상", img
    else:
        cls_label = "손상됨"
        # --- 객체 검출 (YOLOv8) ---
        results = yolo_model.predict(img, imgsz=640)
        det_img = results[0].plot()  # numpy (BGR)
        det_img = Image.fromarray(det_img[..., ::-1])
        return cls_label, det_img

# Gradio UI
demo = gr.Interface(
    fn=pipeline_predict,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Textbox(label="도로 상태"), gr.Image(type="pil", label="탐지 결과")],
    title="도로 상태 분석",
    description="분류 모델에서의 손상 확률이 기준 이상이면 탐지 모델로 파손 부분 탐지",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()