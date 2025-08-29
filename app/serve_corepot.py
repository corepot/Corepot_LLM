from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import torch
import uvicorn
import os
import io

from app.corepot_model import Corepot
from app.utils import load_model  # 通用模型加载函数
from PIL import Image
import torchvision.transforms as transforms

app = FastAPI(title="Corepot通用推理服务")

# -------------------------
# 模型路径与初始化
# -------------------------
model_path = os.environ.get("MODEL_PATH", "models/best_model.pt")
num_classes = int(os.environ.get("NUM_CLASSES", 10))  # 根据训练类别数修改
model = Corepot(num_classes=num_classes)

# 加载模型
if os.path.exists(model_path):
    load_model(model, model_path, map_location="cpu")
    model.eval()
    print(f"Loaded model from {model_path}")
else:
    print(f"Warning: {model_path} not found. 推理服务启动时模型未加载。")

# -------------------------
# 上传路由（可复用原有上传逻辑）
# -------------------------
from app.upload_service import router as upload_router
app.include_router(upload_router)

# -------------------------
# 图像预处理
# -------------------------
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------
# 图像预测接口
# -------------------------
@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    if not os.path.exists(model_path):
        return JSONResponse(content={"error": "模型未加载，无法预测"}, status_code=400)
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = image_transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = output.argmax(dim=1).item()
        return JSONResponse(content={"predicted_class": predicted_class})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# -------------------------
# 文本预测接口（示例）
# -------------------------
@app.post("/predict/text")
async def predict_text(text: str = Form(...)):
    if not os.path.exists(model_path):
        return JSONResponse(content={"error": "模型未加载，无法预测"}, status_code=400)
    try:
        # TODO: 文本处理逻辑，例如 tokenizer -> tensor -> model -> output
        input_tensor = torch.tensor([0])  # 占位
        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = output.argmax(dim=1).item()
        return JSONResponse(content={"predicted_class": predicted_class})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# -------------------------
# 音频预测接口（示例）
# -------------------------
@app.post("/predict/audio")
async def predict_audio(file: UploadFile = File(...)):
    if not os.path.exists(model_path):
        return JSONResponse(content={"error": "模型未加载，无法预测"}, status_code=400)
    try:
        audio_bytes = await file.read()
        # TODO: 音频处理逻辑，例如 librosa -> tensor -> model -> output
        input_tensor = torch.tensor([0])  # 占位
        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = output.argmax(dim=1).item()
        return JSONResponse(content={"predicted_class": predicted_class})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# -------------------------
# 视频预测接口（示例）
# -------------------------
@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    if not os.path.exists(model_path):
        return JSONResponse(content={"error": "模型未加载，无法预测"}, status_code=400)
    try:
        video_bytes = await file.read()
        # TODO: 视频处理逻辑，例如 OpenCV -> 逐帧处理 -> tensor -> model -> output
        input_tensor = torch.tensor([0])  # 占位
        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = output.argmax(dim=1).item()
        return JSONResponse(content={"predicted_class": predicted_class})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# -------------------------
# 启动
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)







