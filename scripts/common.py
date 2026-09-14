"""Tiện ích dùng chung cho mọi script trong scripts/.

Giữ file này nhỏ: chỉ những thứ thực sự lặp lại ở nhiều script mới được vào đây.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def enable_utf8_output() -> None:
    """Cho phép in tiếng Việt trên console Windows.

    Console Windows mặc định dùng cp1252, gặp chữ có dấu là ném UnicodeEncodeError
    và script chết giữa chừng. Gọi hàm này ở đầu mọi script có in tiếng Việt.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


# ── exclusions.csv — nguyên tắc N3: mọi file bị loại đều phải ghi lý do ───────

EXCLUSIONS_PATH = REPO_ROOT / "data" / "manifests" / "exclusions.csv"
EXCLUSION_FIELDS = ["file_id", "stage", "reason_code", "detail", "decided_by", "decided_at"]


def write_exclusions(stage: str, rejected: dict[str, tuple[str, str]], processed: set[str],
                     decided_by: str = "auto") -> None:
    """Ghi các file bị loại ở một giai đoạn, giữ nguyên dòng của giai đoạn khác.

    `rejected`: {file_id: (mã lý do, chi tiết)}
    `processed`: MỌI file đã xét ở lần chạy này — phải xoá hết dòng cũ của chúng ở
        giai đoạn này, không chỉ dòng trùng với lần loại mới. Thiếu điều này thì một
        file từng bị loại rồi nay được nhận (vì ngưỡng đã đổi) sẽ vừa nằm trong
        exclusions.csv vừa nằm trong bank — mâu thuẫn làm dataset không tái lập được.
    """
    import csv
    from datetime import datetime, timezone

    EXCLUSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    kept: list[dict] = []
    if EXCLUSIONS_PATH.exists():
        with EXCLUSIONS_PATH.open(encoding="utf-8", newline="") as handle:
            kept = [r for r in csv.DictReader(handle) if not (r["stage"] == stage and r["file_id"] in processed)]

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new_rows = [
        {"file_id": file_id, "stage": stage, "reason_code": reason,
         "detail": detail, "decided_by": decided_by, "decided_at": timestamp}
        for file_id, (reason, detail) in sorted(rejected.items())
    ]

    with EXCLUSIONS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXCLUSION_FIELDS)
        writer.writeheader()
        writer.writerows(kept + new_rows)
