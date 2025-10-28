from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# 1. 載入模型
with open("pipeline_v1.bin", "rb") as f:
    model = pickle.load(f)

# 2. 定義輸入資料格式
class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

# 3. 建立 FastAPI app
app = FastAPI()

@app.get("/")
def home():
    return {"message": "ML Zoomcamp model is running!"}

@app.post("/predict")
def predict(lead: Lead):
    # 將輸入轉成 dict
    data = lead.dict()
    X = [data]

    # 模型預測轉換機率
    proba = model.predict_proba(X)[0, 1]
    return {"conversion_probability": proba}
