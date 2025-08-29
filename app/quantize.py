import torch
import torch.quantization
import os
from app.corepot_model import Corepot

def quantize_model(model_fp32):
    """
    使用动态量化将模型从 FP32 转为 INT8（适用于 Linear 层）。
    """
    model_fp32.eval()
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32, {torch.nn.Linear}, dtype=torch.qint8
    )
    print("动态量化完成：模型已转为 INT8")
    return model_int8

def save_quantized_model(model, path="models/corepot_quantized.pt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"量化后的模型已保存到 {path}")

def main():
    model = Corepot(num_classes=10)
    checkpoint_path = "models/best_model.pt"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"未找到模型权重: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    quantized_model = quantize_model(model)
    save_quantized_model(quantized_model)

if __name__ == "__main__":
    main()






