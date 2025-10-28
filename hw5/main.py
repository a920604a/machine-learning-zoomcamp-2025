import pickle




model_file = 'pipeline_v1.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# question 3 
def predict_single(record: dict):
    

    #  模型預測（通常 pipeline 包含前處理 + 分類器）
    X = dv.transform([record])
    
    print(f"X :{X}")
    prob = model.predict_proba(X)[0, 1]  # 機率: 轉換為 class=1 的機率

    print(f"Probability of conversion: {prob:.3f}")


def main():
    predict_single(
        {
            "lead_source": "paid_ads",
            "number_of_courses_viewed": 2,
            "annual_income": 79276.0
        }
    )


if __name__ == "__main__":
    main()
