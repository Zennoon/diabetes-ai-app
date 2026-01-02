from enum import Enum
from pydantic import BaseModel, Field


class ModelChoice(Enum):
    LogisticRegression = "logistic"
    DecisionTree = "dt"

class PatientInput(BaseModel):
    pregnancies: int = Field(..., ge=0, description="Number of pregnancies")
    glucose: float = Field(..., ge=0, description="Glucose level")
    bp: float = Field(..., ge=0, description="Blood pressure")
    bmi: float = Field(..., ge=0, description="BMI (Body Mass Index)")
    dpf: float = Field(..., ge=0, description="Diabetes Pedigree Function")
    age: int = Field(..., ge=1)
    model_choice: ModelChoice = ModelChoice.LogisticRegression
