"""Cấu hình đọc từ biến môi trường — không hardcode, không đọc file .env trong container.

Nguyên tắc "validate at system boundaries": biến môi trường LÀ một biên hệ thống. Thiếu
`DATABASE_URL` thì service phải chết ngay lúc khởi động với thông báo rõ, chứ không chạy
được nửa vời rồi ngã ở request đầu tiên.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    database_url: str = "postgresql+asyncpg://asr:asr@db:5432/audio_security"
    redis_url: str = "redis://redis:6379/0"
    inference_url: str = "http://inference:8001"

    # Phải khớp VECTOR(n) trong migration. Đổi model khác số chiều ⇒ viết migration mới.
    embed_dim: int = 1024

    rag_answer_provider: str = "template"
    anthropic_api_key: str = ""
    ollama_url: str = "http://host.docker.internal:11434"
    rag_top_k: int = 5

    # Ngưỡng similarity tối thiểu để một sự kiện được coi là BẰNG CHỨNG.
    #
    # Không có ngưỡng thì `ORDER BY ... LIMIT k` luôn trả về k sự kiện gần nhất dù chúng
    # chẳng liên quan gì, nên nhánh "không có sự kiện nào" của §4.4 không bao giờ chạy
    # được — ràng buộc chống bịa chỉ tồn tại trên giấy. Đo thật: câu "công thức nấu phở
    # bò" từng trả về một sự kiện kính vỡ kèm citation.
    #
    # Hiệu chuẩn trên 12 sự kiện thật + 15 câu hỏi (8 đúng đề, 7 lạc đề), BGE-M3:
    #     đúng đề  0.5561 .. 0.7641   (7/8 trả về đúng sự kiện)
    #     lạc đề   0.2543 .. 0.4903
    # Hai nhóm tách sạch, khoảng cách +0.0658 ⇒ lấy điểm giữa 0.523, làm tròn 0.52.
    #
    # ⚠️ CẦN HIỆU CHUẨN LẠI trên gold_test khi có (W3): 12 sự kiện là quá ít để công bố,
    # và khoảng cách 0.0658 khá hẹp — thêm lớp sự kiện mới có thể làm hai nhóm chồng lấn.
    #
    # ⚠️ HẠN CHẾ ĐÃ BIẾT: câu hỏi gõ KHÔNG DẤU ("co tieng sung khong") chỉ đạt
    # 0.3427..0.3789 — nằm trọn trong vùng lạc đề, nên sẽ bị ngưỡng này từ chối. Người
    # Việt gõ không dấu rất thường xuyên, nên cần khôi phục dấu trước khi nhúng. Ghi lại
    # ở đây vì nó KHÔNG lộ ra nếu chỉ test bằng tiếng Việt có dấu đầy đủ.
    rag_min_similarity: float = 0.52

    # SYSTEM.md §1.4: audio thô có thời hạn; 0 = không lưu audio, chỉ metadata + caption.
    audio_retention_days: int = 30
    audio_dir: Path = Path("/data/event_audio")

    # ontology_map.yaml mount read-only vào container — nguồn chân lý duy nhất cho
    # taxonomy và tier, KHÔNG chép lại 16 lớp vào code service.
    ontology_path: Path = Path("/config/ontology_map.yaml")

    log_level: str = "INFO"
    api_prefix: str = "/api/v1"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
