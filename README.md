Corepot — 深度+推理学习项目

项目简介
Corepot 是一个基于 PyTorch 实现的轻量级图像分类深度学习框架，支持训练、验证、模型剪枝、量化、ONNX 导出和推理服务。

环境依赖
Python 3.8+
PyTorch 1.9+
torchvision
scikit-learn
fastapi
uvicorn
tensorboard

安装依赖：
pip install -r requirements.txt

目录结构
corepot/
├── app/                # 主要代码文件
├── configs/            # 配置文件
├── data/               # 数据集（训练/验证）
├── logs/               # TensorBoard 日志
├── models/             # 模型保存目录
├── scripts/            # 脚本文件（训练、导出、推理启动）
├── tests/              # 测试代码
├── Dockerfile          # Docker 镜像构建文件
├── requirements.txt    # 依赖清单
├── README.md           # 项目说明文件
└── VERSION             # 版本信息


数据准备:
请准备好数据集，放置在 data/train 和 data/val 目录下，目录结构示例：

data/
├── train/
│   ├── cat/
│   ├── dog/
│   └── ...
└── val/
    ├── cat/
    ├── dog/
    └── ...

使用说明
1. 训练模型
bash scripts/train.sh
或直接运行：

python -m app.entrypoint train --data-dir data --epochs 30 --batch-size 64 --lr 0.001
训练过程中会自动保存最佳模型到 models/best_model.pt，并生成 TensorBoard 日志在 logs/runs/。

2. 导出 ONNX 模型

bash scripts/export.sh
或：

python -m app.entrypoint export --output models/corepot.onnx
3. 启动推理服务（FastAPI）

bash scripts/serve.sh
或：
python -m app.entrypoint serve
访问接口：

POST /predict
上传图片，返回分类结果。
4. 模型剪枝

python -m app.prune
剪枝后模型保存在 models/pruned_model.pt。

5. 模型量化

python -m app.quantize
量化后模型保存在 models/quantized_model.pt。

日志与指标
训练过程中，TensorBoard 日志会保存到 logs/runs/，可以用以下命令启动：

tensorboard --logdir logs/runs
然后在浏览器访问 http://localhost:6006 查看训练过程的 Loss、准确率、F1、召回率等指标曲线。

Docker 使用

构建镜像：
docker build -t corepot:latest .

运行训练
docker run --rm -v $(pwd)/data:/app/data corepot:latest train --data-dir data

启动推理服务：
docker run --rm -p 8000:8000 corepot:latest serve

贡献
欢迎提交 Issues 和 Pull Requests，一起完善 Corepot！

许可证
MIT License


