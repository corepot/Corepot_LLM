import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
import os

from app.corepot_model import Corepot

def prune_model(model: nn.Module, amount: float = 0.3) -> nn.Module:
    """
    对模型进行剪枝，默认按比例剪掉每层30%的权重
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
    return model

def save_pruned_model(model: nn.Module, save_path: str = "models/pruned_model.pt"):
    torch.save(model.state_dict(), save_path)
    print(f"剪枝后的模型已保存到: {save_path}")

def main():
    model_path = "models/best_model.pt"
    if not os.path.exists(model_path):
        print(f"未找到模型文件: {model_path}")
        return

    checkpoint = torch.load(model_path, map_location="cpu")
    model = Corepot(num_classes=10)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("开始模型剪枝...")
    pruned_model = prune_model(model, amount=0.3)
    save_pruned_model(pruned_model)

if __name__ == "__main__":
    main()




