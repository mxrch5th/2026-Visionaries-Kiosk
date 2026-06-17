import base64
import io
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "static"
MODEL_PATH = ROOT / "model" / "age_group_resnet18.pt"
CLASS_NAMES = ["under50", "over50"]

app = FastAPI(title="Face Age Cafe Kiosk")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class FramePayload(BaseModel):
    image: str


class AgeModel:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.load()

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if not MODEL_PATH.exists():
            return

        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        state = torch.load(MODEL_PATH, map_location=self.device)
        model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
        model.to(self.device)
        model.eval()
        self.model = model

    def crop_face(self, image: Image.Image) -> tuple[Image.Image, bool]:
        rgb = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return image, False

        x, y, w, h = sorted(faces, key=lambda face: face[2] * face[3], reverse=True)[0]
        pad = int(max(w, h) * 0.18)
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, rgb.shape[1])
        y2 = min(y + h + pad, rgb.shape[0])
        return Image.fromarray(rgb[y1:y2, x1:x2]), True

    def predict(self, image: Image.Image) -> dict:
        face, face_detected = self.crop_face(image)

        if self.model is None:
            # 모델 학습 전에도 화면 기능을 만들고 발표 리허설을 할 수 있게 둔 모의 응답입니다.
            over50_probability = random.uniform(0.08, 0.92)
            return {
                "mode": "easy" if over50_probability >= 0.62 else "normal",
                "label": "over50" if over50_probability >= 0.62 else "under50",
                "confidence": round(over50_probability, 3),
                "faceDetected": face_detected,
                "modelReady": False,
            }

        tensor = self.transform(face).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        over50_probability = float(probabilities[CLASS_NAMES.index("over50")])
        return {
            "mode": "easy" if over50_probability >= 0.62 else "normal",
            "label": CLASS_NAMES[int(np.argmax(probabilities))],
            "confidence": round(over50_probability, 3),
            "faceDetected": face_detected,
            "modelReady": True,
        }


age_model = AgeModel()


def decode_data_url(data_url: str) -> Image.Image:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    image_bytes = base64.b64decode(data_url)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "modelReady": age_model.ready}


@app.post("/api/analyze-frame")
def analyze_frame(payload: FramePayload) -> dict:
    image = decode_data_url(payload.image)
    return age_model.predict(image)
