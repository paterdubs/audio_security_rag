# Image `inference` — nơi DUY NHẤT có torch trong toàn hệ thống.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Cache model vào volume: BGE-M3 ~2.2 GB và checkpoint PANNs ~300 MB, tải lại mỗi
    # lần dựng container là không chấp nhận được.
    HF_HOME=/models/hf \
    PANNS_DATA=/models/panns

WORKDIR /app

# libsndfile: soundfile/librosa cần binary này để đọc wav. Thiếu nó thì lỗi chỉ hiện ra
# lúc chạy, dưới dạng "sndfile library not found" — rất xa nguyên nhân thật.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# panns_inference hardcode ~/panns_data (không đọc biến môi trường nào). Symlink chỗ đó
# vào /models để checkpoint 312 MB rơi vào named volume thay vì nằm trong lớp image.
RUN mkdir -p /models/panns /models/hf && ln -s /models/panns /root/panns_data

COPY services/inference/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/inference/app ./app

EXPOSE 8001
# start-period dài: lần chạy đầu phải tải BGE-M3 (~2.2 GB) và checkpoint PANNs (~300 MB).
# Đặt ngắn thì Docker giết container giữa lúc tải và không bao giờ khởi động được.
HEALTHCHECK --interval=15s --timeout=10s --retries=40 --start-period=120s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8001/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
