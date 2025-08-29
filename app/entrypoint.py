import argparse
from app import train, serve_corepot, export_onnx, prune, quantize

def main():
    parser = argparse.ArgumentParser(description="Corepot 图像分类系统")
    subparsers = parser.add_subparsers(dest="command")

    # 训练子命令
    subparsers.add_parser("train", help="训练模型（使用环境变量配置参数）")

    # 启动推理服务
    subparsers.add_parser("serve", help="启动模型推理服务")

    # 导出 ONNX
    parser_export = subparsers.add_parser("export", help="导出模型为 ONNX")
    parser_export.add_argument("--output", default="models/corepot.onnx", help="ONNX 输出路径")

    # 模型剪枝
    subparsers.add_parser("prune", help="剪枝模型")

    # 模型量化
    subparsers.add_parser("quantize", help="量化模型")

    args = parser.parse_args()

    if args.command == "train":
        # 直接调用 train.train()，参数由环境变量提供
        train.train()
    elif args.command == "serve":
        serve_corepot.serve()
    elif args.command == "export":
        export_onnx.export_model(output_path=args.output)
    elif args.command == "prune":
        prune.prune()
    elif args.command == "quantize":
        quantize.quantize()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
