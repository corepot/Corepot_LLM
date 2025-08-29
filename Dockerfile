# 选择官方 Python 3.9 镜像（轻量）
FROM python:3.9-slim

# 环境变量，避免生成pyc文件，输出日志更及时
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 设置工作目录
WORKDIR /app

# 复制 requirements.txt 并安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制所有代码
COPY . .

# 暴露服务端口（FastAPI 默认8000）
EXPOSE 8000

# 默认执行命令，通过 entrypoint.py 接受参数
ENTRYPOINT ["python", "-m", "app.entrypoint"]
CMD ["serve"]









