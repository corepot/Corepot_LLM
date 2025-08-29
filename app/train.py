import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, recall_score

from app.corepot_model import Corepot
from app.datasets import get_datasets
from app.utils import send_email, save_model, load_model  # 使用新版通用工具

def train():
    # -------------------------
    # 环境变量参数
    # -------------------------
    batch_size = int(os.environ.get("BATCH_SIZE", 64))
    learning_rate = float(os.environ.get("LEARNING_RATE", 0.001))
    epochs = int(os.environ.get("EPOCHS", 10))
    data_dir = os.environ.get("DATA_DIR", "./data")
    save_path = os.environ.get("MODEL_SAVE_PATH", "./models/best_model.pt")
    log_dir = os.environ.get("LOG_DIR", "./logs/runs")

    # 邮件配置（可选）
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    receiver_email = os.getenv("RECEIVER_EMAIL")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # 数据集
    # -------------------------
    train_loader, val_loader = get_datasets(data_dir, batch_size=batch_size)
    num_classes = len(train_loader.dataset.classes)

    # -------------------------
    # 模型、损失函数、优化器
    # -------------------------
    model = Corepot(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # -------------------------
    # 断点续训
    # -------------------------
    start_epoch = 0
    if os.path.exists(save_path):
        checkpoint = load_model(model, save_path, optimizer=optimizer, map_location=device)
        start_epoch = checkpoint.get("epoch", -1) + 1
        print(f"从 epoch {start_epoch} 继续训练...")

    # -------------------------
    # TensorBoard
    # -------------------------
    writer = SummaryWriter(log_dir=log_dir)

    best_val_acc = 0
    epochs_no_improve = 0
    early_stop_patience = 5

    # -------------------------
    # 训练循环
    # -------------------------
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0
        all_preds, all_labels = [], []

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(target.cpu().tolist())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_f1 = f1_score(all_labels, all_preds, average='macro')
        epoch_recall = recall_score(all_labels, all_preds, average='macro')

        # -------------------------
        # 验证
        # -------------------------
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item() * data.size(0)
                preds = output.argmax(dim=1)
                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(target.cpu().tolist())

        val_loss /= len(val_loader.dataset)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        val_recall = recall_score(val_labels, val_preds, average='macro')
        val_acc = sum([p == t for p, t in zip(val_preds, val_labels)]) / len(val_labels)

        # -------------------------
        # TensorBoard 记录
        # -------------------------
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("F1/train", epoch_f1, epoch)
        writer.add_scalar("Recall/train", epoch_recall, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)
        writer.add_scalar("Recall/val", val_recall, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {epoch_loss:.4f}, F1: {epoch_f1:.4f}, Recall: {epoch_recall:.4f} | "
              f"Val Loss: {val_loss:.4f}, F1: {val_f1:.4f}, Recall: {val_recall:.4f}, Acc: {val_acc:.4f}")

        # -------------------------
        # 保存模型
        # -------------------------
        save_model(model, optimizer=optimizer, epoch=epoch, path=save_path)
        print(f"保存模型：{save_path}")

        # -------------------------
        # 提前停止
        # -------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"验证准确率已 {early_stop_patience} 个 epoch 未提升，早停！")
                break

        scheduler.step()

        # -------------------------
        # 训练进度邮件（可选）
        # -------------------------
        if sender_email and sender_password and receiver_email:
            subject = f"Corepot 训练进度 - Epoch {epoch+1}/{epochs}"
            content = (
                f"训练进度:\n"
                f"Epoch {epoch+1}\n"
                f"训练 Loss: {epoch_loss:.4f}, F1: {epoch_f1:.4f}, Recall: {epoch_recall:.4f}\n"
                f"验证 Loss: {val_loss:.4f}, F1: {val_f1:.4f}, Recall: {val_recall:.4f}, Acc: {val_acc:.4f}\n"
            )
            try:
                send_email(subject, content, sender_email, sender_password, receiver_email)
            except Exception as e:
                print(f"邮件发送失败: {e}")

    writer.close()
    print("训练完成")

    # -------------------------
    # 训练完成邮件（可选）
    # -------------------------
    if sender_email and sender_password and receiver_email:
        subject = "Corepot 训练完成"
        content = f"训练完成，最佳验证准确率：{best_val_acc:.4f}"
        try:
            send_email(subject, content, sender_email, sender_password, receiver_email)
        except Exception as e:
            print(f"邮件发送失败: {e}")

if __name__ == "__main__":
    train()


















