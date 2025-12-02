import numpy as np
from PIL import Image
import onnxruntime
from urllib.request import urlopen
from io import BytesIO

# 載入 ONNX 模型
sess = onnxruntime.InferenceSession("./hair_classifier_v1.onnx")
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

def preprocess_image(url):
    with urlopen(url) as resp:
        img = Image.open(BytesIO(resp.read()))
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    img = img.resize((200, 200), Image.NEAREST)
    img_array = np.array(img).astype(np.float32) / 255.0

    # 使用 float32 的 mean/std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img_array = (img_array - mean) / std
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, 0)

    # 確保整個 array 是 float32
    return img_array.astype(np.float32)

def handler(event=None, context=None):
    """
    Lambda handler function
    event: dict，需包含 "url" key
    """
    url = event.get("url") if event and "url" in event else \
        "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
    
    img_array = preprocess_image(url)
    
    # 模型推論
    output = sess.run([output_name], {input_name: img_array})[0]
    
    # 回傳 JSON 友善格式
    return {"output": output.tolist()}

# 本地測試
if __name__ == "__main__":
    test_event = {
        "url": "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
    }
    result = handler(test_event, None)
    print(result)
