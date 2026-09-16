# Image `api` — CỐ Ý không có torch. Mọi thứ cần model nằm ở service `inference`.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Cài dependency TRƯỚC khi copy code: tầng này chỉ build lại khi requirements đổi,
# nên sửa code không phải cài lại toàn bộ thư viện.
COPY services/api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/api/app ./app
COPY services/api/alembic ./alembic
COPY services/api/alembic.ini ./alembic.ini
COPY services/api/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

EXPOSE 8000
# Healthcheck gọi đúng endpoint thật (có kiểm tra database), không phải cổng TCP suông:
# cổng mở không có nghĩa là service phục vụ được.
HEALTHCHECK --interval=10s --timeout=5s --retries=12 --start-period=20s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/v1/health').status==200 else 1)"

ENTRYPOINT ["./entrypoint.sh"]
