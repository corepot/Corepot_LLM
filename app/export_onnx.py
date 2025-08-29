import torch
import os
from app.corepot_model import Corepot

def export_to_onnx(model, dummy_input, output_path="models/corepot.onnx"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"模型已成功导出为 ONNX 格式：{output_path}")

def main():
    model = Corepot(num_classes=10)
    checkpoint_path = "models/best_model.pt"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"未找到模型权重: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    export_to_onnx(model, dummy_input)

if __name__ == "__main__":
    main()








