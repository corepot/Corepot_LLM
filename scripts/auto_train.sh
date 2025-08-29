#!/bin/bash

# 1. 设置环境变量
export DATA_DIR=/app/data
export MODEL_DIR=/app/models
export LOG_DIR=/app/logs/runs

echo "==== 自动训练开始 $(date) ===="

# 2. (可选) 同步最新数据，比如从远程服务器或云盘
# rsync -avz user@remote:/path/to/data $DATA_DIR

# 3. 运行训练脚本
python -m app.entrypoint train --data-dir $DATA_DIR --epochs 20 --batch-size 64 --lr 0.001 --log-dir $LOG_DIR --save-path $MODEL_DIR/best_model.pt

# 4. 检查训练结果
if [ $? -eq 0 ]; then
    echo "训练成功，模型已保存到 $MODEL_DIR/best_model.pt"
else
    echo "训练失败，请检查日志"
fi

echo "==== 自动训练结束 $(date) ===="




