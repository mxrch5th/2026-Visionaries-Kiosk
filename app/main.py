import os
import base64
import io
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "static"
MODEL_PATH = ROOT / "model" / "best_model_v3_final.pth"
DEFAULT_CLASS_NAMES = ["under50", "over50"]

app = FastAPI(title="Face Age Cafe Kiosk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class FramePayload(BaseModel):
    image: str

class AgeModel:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = DEFAULT_CLASS_NAMES
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.load()

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if not MODEL_PATH.exists():
            print(f"⚠️ 모델 파일 없음: {MODEL_PATH}")
            return
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=self.device)
            state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
            
            self.model = models.mobilenet_v2(weights=None)
            self.model.classifier[1] = nn.Linear(self.model.last_channel, 2)
            
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print("✨ [성공] AI 모델 탑재 완료!")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")

    def predict(self, pil_img: Image.Image) -> dict:
        if not self.ready:
            return {"prediction": "under50", "confidence": 0.0, "probability": 0.0}
        
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            
            over50_prob = float(probs[1])  # 50대 이상 확률
            pred_idx = probs.argmax().item()
            
        # 친구 프론트 자바스크립트가 어떤 변수명을 쓰든 다 매칭되게 뚫어놓음
        return {
            "prediction": self.class_names[pred_idx],
            "confidence": over50_prob,
            "probability": over50_prob,
            "prob": over50_prob,
            "over50_probability": over50_prob,
            "percent": over50_prob * 100,
            "percentage": over50_prob * 100
        }

age_model = AgeModel()

# 터미널 로그에 찍힌 대로 원래 친구 주소 규격인 /predict 로 복구
@app.post("/predict")
async def predict_age(payload: FramePayload):
    try:
        img_data = base64.b64decode(payload.image.split(",")[-1])
        pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")
        
        open_cv_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        faces = age_model.face_detector.detectMultiScale(gray, 1.1, 4)
        
        # 얼굴 안 잡혀도 NaN 안 뜨고 0% 강제 리턴하도록 안전장치
        if len(faces) == 0:
            return {
                "prediction": "under50",
                "confidence": 0.0,
                "probability": 0.0,
                "prob": 0.0,
                "over50_probability": 0.0,
                "percent": 0.0,
                "percentage": 0.0,
                "message": "얼굴 미감지"
            }
            
        x, y, w, h = faces[0]
        pad_w, pad_h = int(w * 0.15), int(h * 0.15)
        img_h, img_w, _ = open_cv_image.shape
        cropped = open_cv_image[max(0, y-pad_h):min(img_h, y+h+pad_h), max(0, x-pad_w):min(img_w, x+w+pad_w)]
        pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

        return age_model.predict(pil_img)
    except Exception as e:
        return {
            "prediction": "under50", "confidence": 0.0, "probability": 0.0, 
            "prob": 0.0, "percent": 0.0, "error": str(e)
        }

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
