import joblib
import pandas as pd
import numpy as np

from models.schemas import ModelChoice

class DiabetesService:
    def __init__(self):
        self.lr_pipeline = joblib.load("models/logistic_model.joblib")
        self.dt_pipeline = joblib.load("models/decision_tree_model.joblib")
        try:
            self.background_df = pd.read_csv("models/train_data.csv")
        except FileNotFoundError:
            self.background_df = pd.DataFrame()
    
    def get_background_data(self):
        return self.background_df.to_dict(orient="records")

    def predict_single(self, data: dict):
        model_choice: ModelChoice = data.pop("model_choice")
        mapping = {
            "pregnancies": "Pregnancies",
            "glucose": "Glucose",
            "bp": "BloodPressure",
            "bmi": "BMI",
            "dpf": "DiabetesPedigreeFunction",
            "age": "Age"
        }
        raw_df = pd.DataFrame([{ mapping[key]: value for key, value in data.items()}])
        if model_choice == ModelChoice.LogisticRegression:
            raw_df["BMI_Age_Interaction"] = raw_df["BMI"] * raw_df["Age"]
            model = self.lr_pipeline
        else:
            model = self.dt_pipeline
        
        prediction = model.predict(raw_df)[0]
        probability = model.predict_proba(raw_df)[0][1]

        return {
            "prediction": int(prediction),
            "probability": round(float(probability)*100, 2),
            "risk_level": "High" if probability > 0.5 else "Low"
        }
    
    def predict_batch(self, df: pd.DataFrame, model_choice: ModelChoice):
        working_df = df.copy()
        if model_choice == ModelChoice.LogisticRegression:
            working_df["BMI_Age_Interaction"] = working_df["BMI"] * working_df["Age"]
            model = self.lr_pipeline
        else:
            model = self.dt_pipeline
        
        preds = model.predict(working_df)
        probs = model.predict_proba(working_df)[:, 1]

        df["Prediction"] = ["Diabetic" if p == 1 else "Healthy" for p in preds]
        df["Probability %"] = (probs * 100).round(2)
        
        return df.to_dict(orient="records")