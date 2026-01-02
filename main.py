from fastapi import FastAPI, Request, UploadFile, File, Form, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from models.schemas import ModelChoice, PatientInput
from services.predictor import DiabetesService
import pandas as pd
import io

app = FastAPI(title="AI Diabetes Diagnosis")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

service = DiabetesService()

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/background-data")
async def background_data():
    return service.get_background_data()

@app.post("/api/predict")
async def predict(data: PatientInput):
    return service.predict_single(data.model_dump())

@app.post("/api/batch-predict")
async def batch_predict(file: UploadFile = File(...), model_choice: ModelChoice = Form()):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    results = service.predict_batch(df, model_choice)
    return { "results": results }

@app.get("/health")
def health_check():
    return {"status": "awake"}