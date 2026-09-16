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
