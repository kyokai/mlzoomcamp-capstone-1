import bentoml
from bentoml.io import JSON
from pydantic import BaseModel


class DiabetesAssessment(BaseModel):
    pregnancies: int = 1.0
    glucose: int = 151.0
    bloodpressure: int = 60.0
    skinthickness: int = 0.0
    insulin: int = 0.0
    bmi: float = 26.1
    diabetespedigreefunction: float = 0.179
    age: int = 22.0


model_ref = bentoml.xgboost.get("diabetes_risk_model:latest")
dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()

svc = bentoml.Service("diabetes_risk_classifier", runners=[model_runner])


@svc.api(input=JSON(pydantic_model=DiabetesAssessment), output=JSON())
def classify(raw_data):
    vector = dv.transform(raw_data.dict())
    prediction = model_runner.predict.run(vector)
    result = prediction[0]

    if result > 0.7:
        return {
            "status": 200,
            "probability": result,
            "diabetes_risk": "HIGH"
        }
    elif result > 0.5:
        return {
            "status": 200,
            "probability": result,
            "diabetes_risk": "MEDIUM"
        }
    elif result > 0.25:
        return {
            "status": 200,
            "probability": result,
            "diabetes_risk": "LOW"
        }
    else:
        return {
            "status": 200,
            "probability": result,
            "diabetes_risk": "UNLIKELY"
        }
