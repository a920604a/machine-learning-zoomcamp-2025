FROM agrigorev/model-2025-hairstyle:v1

# 安裝必要套件
RUN python3.13 -m pip install --no-cache-dir pillow onnxruntime numpy

# 複製 lambda handler
COPY lambda_handler.py .

# 設定 Lambda handler
CMD ["lambda_handler.handler"]
