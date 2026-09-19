# Grounded Automated Audio Captioning for Security Surveillance and RAG-based Alert Retrieval

Khoá luận tốt nghiệp — Khoa học dữ liệu, IUH K18.

> Đối soát 19/09/2026: [trạng thái và bằng chứng mới nhất](docs/STATUS.md).
> Hiện chạy được **upload offline → PANNs pretrained → caption template → RAG + dashboard**.
> Streaming và Grounded AAC nghiên cứu mô tả dưới đây là mục tiêu, chưa triển khai đầy đủ.
> Baseline nghiên cứu PANNs v1/v2/v3 đã được chấm lại trên cùng một tập dev có quét
> ngưỡng (19/09) — xem [STATUS §3](docs/STATUS.md). Vẫn **chưa có real dev/gold** và chưa deploy
> checkpoint v3 vào serving, nên không đọc đây là kết quả cuối: dev tổng hợp sinh cùng recipe
> với train của v3 nên thiên vị v3 theo thiết kế.

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
| [docs/TRAINING_OPS_PLAN.md](docs/TRAINING_OPS_PLAN.md) | Kế hoạch tracking, kiểm tra dữ liệu và phân tích lỗi |
| [docs/RELATED_WORK_2026.md](docs/RELATED_WORK_2026.md) | Văn liệu, nguồn gốc và giới hạn so sánh |

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
`scaper` dùng `np.Inf` (đã bị gỡ ở numpy 2.0). Cài thêm gói không giữ constraints có
thể nâng NumPy và làm hỏng Scaper; không phải cứ cài FastAPI/Torch là bắt buộc lên 2.x.
Đã có sự cố môi trường trôi phiên bản ngày 14/09/2026.

| Môi trường | Dùng cho | Cài bằng |
|---|---|---|
| `.venv` | `scripts/` và baseline SED PANNs hiện tại | `requirements-data.txt` + `requirements-screen.txt`; snapshot training còn có dependency riêng, chưa có lock hoàn chỉnh |
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

# Test service phải chạy TÁCH từng bộ. Gộp một lệnh `pytest services` sẽ chết với
# `ModuleNotFoundError: No module named 'tests.test_tagger'`: cả hai service đều có gói
# `tests`, pytest nạp gói đầu rồi tưởng gói sau cũng nằm trong đó.
PYTHONPATH=services/api      .venv-services/Scripts/python.exe -m pytest services/api/tests -q
PYTHONPATH=services/inference .venv-services/Scripts/python.exe -m pytest services/inference/tests -q

# Dashboard (cần Node 20+)
cd services/frontend && npm ci && npm run lint && npm test
```

---

## Cấu trúc

```
data/        dữ liệu (đa phần git-ignore; DVC pipeline/remote chưa thiết lập)
docs/        đặc tả, kế hoạch, taxonomy, ADR
ml/          config, dataset, model, training, evaluation
scripts/     pipeline dữ liệu (tải → chuẩn hoá → sàng lọc → chia tập → sinh Scaper)
services/    api · inference · frontend  (stream: W7)
tests/       test cho scripts/
```

> Khi chuẩn bị push, **không** dùng `git add .`: checkpoint `.pt/.pth` và cache waveform là artifact
> cục bộ lớn; JAMS legacy là bằng chứng trước-sửa cần được quyết định theo dõi có chủ đích. Xem STATUS.

`data/synthetic_legacy/` được theo dõi để audit nhãn trước-sửa. Các JAMS này giữ nguyên đường dẫn
tuyệt đối từ máy đã sinh dữ liệu, nên không dùng để replay trực tiếp trên máy mới; code/config/manifest
hiện hành mới là nguồn để tái tạo dữ liệu portable. Weight model phải tải hoặc train lại tại máy clone.

## Giấy phép & đạo đức

Audio từ các nguồn công khai giữ **license theo từng clip** trong
`data/manifests/raw_manifest.csv`; phần lớn **không được phát hành lại**, chỉ công bố nhãn.
Hệ thống **không** nhận dạng người nói và **không** lưu nội dung lời nói dạng văn bản; audio
thô có cấu hình dự kiến thời hạn lưu (`AUDIO_RETENTION_DAYS`, mặc định 30 ngày),
nhưng **chưa có tác vụ tự xoá khi hết hạn**. Không coi đây là retention đã thực thi.
Xem `docs/SYSTEM.md` §1.4.
