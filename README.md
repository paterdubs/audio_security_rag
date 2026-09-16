# Grounded Automated Audio Captioning for Security Surveillance and RAG-based Alert Retrieval

Khoá luận tốt nghiệp — Khoa học dữ liệu, IUH K18.

Hệ thống nghe luồng âm thanh liên tục → phát hiện sự kiện an ninh có định vị thời gian →
sinh mô tả ngôn ngữ tự nhiên **được ràng buộc vào bằng chứng âm học** → đánh giá rủi ro và
cảnh báo → lưu trữ để truy vấn lịch sử bằng tiếng Việt qua RAG **có trích dẫn**.

| Tài liệu | Nội dung |
|---|---|
| [CLAUDE.md](CLAUDE.md) | Trạng thái hiện tại, việc tiếp theo — **đọc đầu tiên** |
| [docs/SYSTEM.md](docs/SYSTEM.md) | Đặc tả hệ thống đầy đủ |
| [docs/PLAN.md](docs/PLAN.md) | Kế hoạch 8 tuần + tiêu chí nghiệm thu |
| [docs/DATA_PLAN.md](docs/DATA_PLAN.md) | Quy trình chuẩn bị dữ liệu |
| [docs/taxonomy.md](docs/taxonomy.md) | Định nghĩa 16 lớp âm thanh |
| [docs/data_inventory.md](docs/data_inventory.md) | Tổng kết dữ liệu (sinh tự động) |

---

## Chạy hệ thống

```bash
cp .env.example .env          # sửa POSTGRES_PASSWORD trước khi chạy
docker compose up -d --build
docker compose ps             # mọi service phải ở trạng thái healthy
```

| Service | URL |
|---|---|
| Dashboard | http://localhost:3000 |
| API (OpenAPI docs) | http://localhost:8000/docs |
| Inference | http://localhost:8001/health |
| MLflow | http://localhost:5000 |

Lần chạy đầu tải BGE-M3 (~2.2 GB) và checkpoint PANNs (~300 MB) vào named volume —
chậm một lần, các lần sau dùng lại.

---

## Hai môi trường Python, cố ý tách rời

**Đây không phải sự bất tiện thừa.** `requirements-data.txt` ghim `numpy==1.26.4` vì
`scaper` dùng `np.Inf` (đã bị gỡ ở numpy 2.0). Cài `fastapi`/`torch` mới vào cùng môi
trường sẽ kéo numpy 2.x về và **làm hỏng toàn bộ bước sinh dữ liệu huấn luyện** — đã xảy
ra thật ngày 14/09/2026.

| Môi trường | Dùng cho | Cài bằng |
|---|---|---|
| `.venv` | `scripts/` — pipeline dữ liệu | `pip install -r requirements-data.txt` (+ `requirements-screen.txt` nếu chạy sàng lọc PANNs) |
| `.venv-services` | `services/` — test logic thuần | `pip install -r requirements-dev.txt` |
| Docker image | `services/` khi chạy thật | mỗi service một `requirements.txt` riêng |

> ⚠️ Trên Windows/Git Bash phải gọi `.venv/Scripts/python.exe`, **không** gọi `python` trần —
> `python` trỏ vào Python hệ thống và script chết ngay ở `import yaml`, trông hệt như "tự dừng".

---

## Kiểm thử

```bash
.venv/Scripts/python.exe -m pytest                  # test pipeline dữ liệu
.venv/Scripts/python.exe scripts/verify_ontology.py # taxonomy ↔ ontology khớp nhau
.venv/Scripts/python.exe scripts/check_leakage.py   # không rò rỉ giữa các tập

.venv-services/Scripts/python.exe -m pytest services   # test logic của service
```

---

## Cấu trúc

```
data/        dữ liệu (DVC quản; git bỏ qua trừ manifests/reference/gold)
docs/        đặc tả, kế hoạch, taxonomy, ADR
ml/          config, dataset, model, training, evaluation
scripts/     pipeline dữ liệu (tải → chuẩn hoá → sàng lọc → chia tập → sinh Scaper)
services/    api · inference · frontend  (stream: W7)
tests/       test cho scripts/
```

## Giấy phép & đạo đức

Audio từ các nguồn công khai giữ **license theo từng clip** trong
`data/manifests/raw_manifest.csv`; phần lớn **không được phát hành lại**, chỉ công bố nhãn.
Hệ thống **không** nhận dạng người nói và **không** lưu nội dung lời nói dạng văn bản; audio
thô có thời hạn lưu trữ (`AUDIO_RETENTION_DAYS`, mặc định 30 ngày). Xem `docs/SYSTEM.md` §1.4.
