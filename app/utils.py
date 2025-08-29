import smtplib
from email.mime.text import MIMEText
from email.header import Header
import torch
import os

# ------------------------------
# 邮件发送工具
# ------------------------------
def send_email(subject, content, sender_email, sender_password, receiver_email,
               smtp_server="smtp.gmail.com", smtp_port=587):
    """
    发送邮件通知

    参数：
    - subject: 邮件主题
    - content: 邮件正文（文本）
    - sender_email: 发件人邮箱
    - sender_password: 发件人邮箱密码或授权码
    - receiver_email: 收件人邮箱
    - smtp_server: SMTP服务器地址（默认为 Gmail）
    - smtp_port: SMTP端口（Gmail使用587）
    """
    msg = MIMEText(content, 'plain', 'utf-8')
    msg['Subject'] = Header(subject, 'utf-8')
    msg['From'] = sender_email
    msg['To'] = receiver_email
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, [receiver_email], msg.as_string())
        server.quit()
        print("邮件发送成功")
    except Exception as e:
        print(f"邮件发送失败: {e}")

# ------------------------------
# 通用模型保存工具
# ------------------------------
def save_model(model, optimizer=None, epoch=None, path="models/best_model.pt"):
    """
    保存模型，可用于训练断点续训或推理

    - model: PyTorch 模型
    - optimizer: 可选，优化器
    - epoch: 可选，当前训练轮数
    - path: 模型保存路径
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict()
    }
    if optimizer:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    torch.save(checkpoint, path)
    print(f"模型已保存到 {path}")

# ------------------------------
# 通用模型加载工具
# ------------------------------
def load_model(model, path="models/best_model.pt", optimizer=None, map_location=None, strict=True):
    """
    加载模型，兼容断点续训和推理，同时自动处理输出层类别不匹配的问题

    参数：
    - model: PyTorch 模型
    - path: 模型路径
    - optimizer: 可选，优化器
    - map_location: 可选，加载设备
    - strict: 是否严格匹配 state_dict，默认 True；如果输出层类别不一致可设置为 False

    返回：checkpoint dict 或 None
    """
    if not os.path.exists(path):
        print(f"Warning: {path} 不存在，模型未加载")
        return None

    checkpoint = torch.load(path, map_location=map_location)

    # 获取 state_dict，兼容旧模型
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # 处理输出层 size mismatch
    model_state_dict = model.state_dict()
    for k in state_dict.keys():
        if k in model_state_dict and state_dict[k].shape != model_state_dict[k].shape:
            print(f"Warning: 参数 {k} 尺寸不匹配，跳过加载")
            state_dict[k] = model_state_dict[k]

    # 加载权重
    model.load_state_dict(state_dict, strict=False if not strict else strict)

    # 加载优化器状态（如果提供）
    if optimizer and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            print(f"Warning: 无法加载优化器状态，可能因为模型结构改变: {e}")

    print(f"模型已加载: {path}")
    return checkpoint







