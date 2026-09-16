#!/bin/sh
# Migrate roi moi phuc vu — SYSTEM.md §9.5: `docker compose up` tu may trang phai tu
# dong cho healthcheck, migrate, seed roi moi nhan request.
#
# ASCII only: file nay chay bang /bin/sh trong container, tranh moi rui ro encoding.
set -e

echo "[entrypoint] chay migration..."
alembic upgrade head

echo "[entrypoint] khoi dong uvicorn..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --proxy-headers
