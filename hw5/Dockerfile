# 1️⃣ 使用提供的 base image
FROM agrigorev/zoomcamp-model:2025

# 2️⃣ 設定工作目錄
WORKDIR /code

# 3️⃣ 複製專案檔案到容器
COPY serve.py ./  
COPY pipeline_v1.bin ./  

# 如果有其他必要檔案，例如 pipeline_v2.bin 可以先確認 base image 已包含

# 4️⃣ 安裝必要套件 (FastAPI + uvicorn + pyproject.toml 依賴)
RUN pip install --no-cache-dir fastapi uvicorn scikit-learn==1.6.1

# 5️⃣ 開放端口
EXPOSE 8000

# 6️⃣ 啟動 FastAPI server
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
