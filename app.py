import joblib
from pathlib import Path
import numpy as np
from datetime import datetime
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Breast Cancer Triage Service")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load(BASE_DIR / "model_svm.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")

class BiopsyInput(BaseModel):
    sample_id: str = Field(..., example="PAT-101")
    features: List[float]

class PredictionOutput(BaseModel):
    sample_id: str
    diagnosis: str
    malignant_probability: float
    priority_review_flag: bool
    clinical_recommendation: str
    timestamp: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def serve_frontend():
    return FileResponse(BASE_DIR / "index.html")


@app.post("/predict", response_model=PredictionOutput)
def predict_biopsy(data: BiopsyInput):
    if len(data.features) != 30:
        raise HTTPException(status_code=400, detail=f"Expected 30 features, got {len(data.features)}")

    scaled = scaler.transform(np.array(data.features).reshape(1, -1))
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]
    is_mal = bool(pred == 1)

    return PredictionOutput(
        sample_id=data.sample_id,
        diagnosis="Malignant" if is_mal else "Benign",
        malignant_probability=round(float(prob), 4),
        priority_review_flag=is_mal,
        clinical_recommendation=(
            "URGENT: High probability of malignancy. Queued for immediate priority review."
            if is_mal else
            "ROUTINE: Biopsy exhibits benign characteristics. Standard review."
        ),
        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    )