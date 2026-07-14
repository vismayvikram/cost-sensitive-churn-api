from pydantic import BaseModel, Field
from typing import List, Literal


class CustomerFeatures(BaseModel):
    """
    Raw, human-readable customer fields — matches the original Telco
    Churn CSV columns. The API handles one-hot encoding internally,
    so callers never need to know about gender_Male, Contract_One year,
    etc.
    """
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, description="Months as a customer")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]
    MonthlyCharges: float = Field(..., gt=0)
    TotalCharges: float = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 2,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.5,
                "TotalCharges": 171.0,
            }
        }


class ReasonCode(BaseModel):
    feature: str
    value: float
    impact: float
    direction: Literal["increases", "decreases"]


class ChurnPrediction(BaseModel):
    churn_probability: float
    will_churn_flag: bool
    threshold_used: float
    fn_cost_if_missed: float = Field(
        ..., description="Estimated $ lost if this churner is missed (MonthlyCharges x 12)"
    )
    fp_cost_if_wrong: float = Field(
        ..., description="Cost of an unnecessary retention offer if flagged incorrectly"
    )
    top_reasons: List[ReasonCode]