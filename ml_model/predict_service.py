from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("mode/hero_model.pkl")

class MatchFeatures(BaseModel):
    radiant_heroes: list[int]
    dire_heroes: list[int]
    rank_tier: int

@app.post("/predict")
def predict(data: MatchFeatures):
    features = data.radiant_heroes + data.dire_heroes + [data.rank_tier]
    predction = model.predict([features])
    return {"Recommended hero id : ": int(predction[0])}