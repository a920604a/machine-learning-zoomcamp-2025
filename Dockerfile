# 使用官方 Python 3.11 slim 版當基底
FROM python:3.11-slim

# 安裝基本系統工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# 建立工作目錄
WORKDIR /workspace

# 安裝常用 Python 套件
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    jupyter \
    torch==2.8.0 \
    torchsummary \
    torchvision \
    onnx \
    onnxruntime

# 預設啟動 Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
