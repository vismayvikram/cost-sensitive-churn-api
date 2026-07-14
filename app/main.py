from fastapi import FastAPI, HTTPException

from fastapi.responses import HTMLResponse
import os

from app.schemas import CustomerFeatures, ChurnPrediction
from app.model import ChurnModel

app = FastAPI(
    title="Cost-Sensitive Churn Prediction API",
    description=(
        "Predicts churn risk using a threshold chosen to minimize expected "
        "business cost (not accuracy/F1), with per-customer SHAP explanations."
    ),
    version="1.0.0",
)

# Loaded once at startup, reused across requests
churn_model = ChurnModel()


@app.get("/", response_class=HTMLResponse)
def root():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dashboard template not found")


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=ChurnPrediction)
def predict_churn(customer: CustomerFeatures):
    try:
        return churn_model.predict(customer.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))