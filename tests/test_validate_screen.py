"""Test cho scripts/validate_screen.py.

Script này sinh ra con số DUY NHẤT biện minh cho việc tự động hoá bước sàng lọc. Nếu
nó sai thì cả bank được tuyên bố là sạch dựa trên một phép đo sai — và không có cổng
nào phía sau bắt được, vì mọi bước sau đều tin vào bank.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import validate_screen as vs  # noqa: E402


def _clip(file_id: str, class_id: str, decision: str = "auto_accept") -> dict:
    return {"file_id": file_id, "class_id": class_id, "decision": decision,
            "p_target": "0.8", "p_confusable": "0.1", "top_confusable": "",
            "onset": "0.0", "offset": "1.0", "path_norm": f"x/{file_id}.wav"}


def _audited(verdict: str, class_id: str = "siren") -> dict:
    return {**_clip(f"f_{verdict}_{class_id}", class_id), "verdict": verdict, "note": ""}


# ── Rút mẫu phân tầng ────────────────────────────────────────────────────────


def test_lop_nho_van_co_mat_trong_mau():
    """Rút toàn cục 10% rất dễ bỏ sót hẳn một lớp nhỏ.

    Mà lớp nhỏ và lớp dễ nhầm mới là nơi bộ sàng lọc sai nhiều nhất — một tỉ lệ lỗi
    tổng thể đẹp đẽ do `speech_normal` đóng góp toàn bộ thì không nói lên điều gì.
    """
    clips = [_clip(f"big{i}", "speech_normal") for i in range(500)]
    clips += [_clip(f"small{i}", "fireworks") for i in range(3)]
    mau = vs.stratified_sample(clips, fraction=0.10, seed=1)
    assert any(r["class_id"] == "fireworks" for r in mau)


def test_moi_lop_lay_it_nhat_mot_clip():
    clips = [_clip("a", "siren"), _clip("b", "gunshot")]
    mau = vs.stratified_sample(clips, fraction=0.10, seed=1)
    assert {r["class_id"] for r in mau} == {"siren", "gunshot"}


def test_ti_le_lay_dung_khoang_10_phan_tram():
    clips = [_clip(f"f{i}", "siren") for i in range(200)]
    assert len(vs.stratified_sample(clips, fraction=0.10, seed=1)) == 20


def test_khong_lay_qua_so_clip_hien_co():
    clips = [_clip("a", "siren")]
    assert len(vs.stratified_sample(clips, fraction=0.50, seed=1)) == 1


def test_cung_seed_cho_cung_mau():
    clips = [_clip(f"f{i}", "siren") for i in range(100)]
    a = [r["file_id"] for r in vs.stratified_sample(clips, 0.10, seed=7)]
    b = [r["file_id"] for r in vs.stratified_sample(clips, 0.10, seed=7)]
    assert a == b


def test_doi_seed_cho_mau_khac():
    clips = [_clip(f"f{i}", "siren") for i in range(100)]
    a = [r["file_id"] for r in vs.stratified_sample(clips, 0.10, seed=7)]
    b = [r["file_id"] for r in vs.stratified_sample(clips, 0.10, seed=8)]
    assert a != b


def test_chi_rut_tu_clip_TU_DONG_NHAN():
    # Kiểm định là về auto-accept. Trộn clip người đã duyệt vào sẽ làm tỉ lệ lỗi đẹp
    # lên một cách giả tạo, vì clip người duyệt gần như luôn đúng.
    clips = [_clip("a", "siren"), _clip("b", "siren", decision="review"),
             _clip("c", "siren", decision="auto_reject")]
    sheet = vs.build_audit_sheet(clips, fraction=1.0, seed=1)
    assert [r["file_id"] for r in sheet] == ["a"]


def test_chua_co_clip_nao_duoc_nhan_thi_dung():
    with pytest.raises(SystemExit):
        vs.build_audit_sheet([_clip("a", "siren", decision="review")], 0.10, seed=1)


def test_phieu_co_cot_verdict_de_trong_cho_nguoi_dien():
    sheet = vs.build_audit_sheet([_clip("a", "siren")], fraction=1.0, seed=1)
    assert sheet[0]["verdict"] == ""


# ── Tính tỉ lệ lỗi ───────────────────────────────────────────────────────────


def test_audio_hong_khong_tinh_la_loi_sang_loc():
    """Bộ sàng lọc trả lời "clip này có đúng lớp không", không phải "audio có tốt không".

    Gộp chung sẽ làm tỉ lệ lỗi phồng lên và đẩy ta đi nâng ngưỡng τ để chữa một bệnh
    thuộc về bước chuẩn hoá.
    """
    audited = [_audited("ok"), _audited("bad_audio", "gunshot"), _audited("wrong_class", "fireworks")]
    errors, total, _ = vs.error_rate(audited)
    assert (errors, total) == (1, 3)


def test_dong_chua_dien_thi_khong_tinh_vao_mau_so():
    # Tính dòng trống là "đúng" sẽ làm tỉ lệ lỗi thấp giả tạo theo đúng số clip chưa nghe.
    audited = [_audited("ok"), {**_clip("b", "siren"), "verdict": "", "note": ""}]
    errors, total, missing = vs.error_rate(audited)
    assert (errors, total, len(missing)) == (0, 1, 1)


def test_gia_tri_verdict_la_thi_coi_nhu_chua_dien():
    audited = [{**_clip("a", "siren"), "verdict": "hình như đúng", "note": ""}]
    _, total, missing = vs.error_rate(audited)
    assert total == 0 and len(missing) == 1


# ── Khoảng tin cậy ───────────────────────────────────────────────────────────


def test_khong_bao_gio_cho_can_duoi_am():
    """Cỡ mẫu nhỏ + tỉ lệ gần 0 là đúng hai điều kiện làm công thức chuẩn cho cận âm.

    Một khoảng [-2%, 8%] in vào khoá luận là tự tố cáo mình dùng sai công cụ.
    """
    for errors, total in [(0, 5), (0, 20), (1, 10), (0, 100)]:
        low, high = vs.wilson_interval(errors, total)
        assert low >= 0.0 and high <= 1.0


def test_khong_thay_loi_nao_van_khong_ket_luan_duoc_la_0():
    # 0/20 KHÔNG có nghĩa tỉ lệ lỗi là 0 — cận trên vẫn trên 10%, tức mẫu chưa đủ.
    low, high = vs.wilson_interval(0, 20)
    assert low == 0.0 and high > 0.10


def test_mau_cang_lon_khoang_cang_hep():
    _, hep = vs.wilson_interval(5, 500)
    _, rong = vs.wilson_interval(1, 10)
    assert hep < rong


def test_mau_rong_thi_tra_ve_khoang_rong_nhat_co_the():
    assert vs.wilson_interval(0, 0) == (0.0, 1.0)


def test_toan_bo_deu_sai_thi_can_tren_gan_1():
    low, high = vs.wilson_interval(10, 10)
    assert high > 0.9 and low < 1.0


# ── Cổng chấp nhận ───────────────────────────────────────────────────────────


def test_vuot_ngan_sach_loi_thi_bao_khong_dat(capsys):
    audited = [_audited("wrong_class") for _ in range(3)] + [_audited("ok") for _ in range(7)]
    errors, total, _ = vs.error_rate(audited)
    assert vs.report(errors, total, audited) is False
    assert "NÂNG ngưỡng" in capsys.readouterr().out


def test_duoi_ngan_sach_va_mau_du_lon_thi_dat(capsys):
    audited = [_audited("ok") for _ in range(300)]
    errors, total, _ = vs.error_rate(audited)
    assert vs.report(errors, total, audited) is True
    assert "đạt" in capsys.readouterr().out


def test_ti_le_dep_nhung_mau_qua_nho_thi_van_canh_bao(capsys):
    # 0/20 → tỉ lệ 0% nhưng cận trên 16%. Báo "đạt" trần trụi ở đây là kết luận sai.
    audited = [_audited("ok") for _ in range(20)]
    errors, total, _ = vs.error_rate(audited)
    vs.report(errors, total, audited)
    assert "cận trên" in capsys.readouterr().out


# ── Phiếu duyệt phải MÙ ──────────────────────────────────────────────────────


def test_phieu_khong_chua_diem_cua_may():
    """Dặn người duyệt nghe mù rồi in sẵn p_target ngay cạnh là lời dặn vô nghĩa.

    Biết máy chấm 0.95 rồi mới nghe thì kết luận đã bị neo vào đó, và phép kiểm định
    chỉ còn đo được mức độ người đồng ý với máy khi đã biết máy nghĩ gì — không đo
    được điều ta cần, là máy có đúng hay không.
    """
    cam = {"p_target", "p_confusable", "top_confusable", "decision", "priority"}
    assert not (cam & set(vs.AUDIT_FIELDS))


def test_phieu_co_du_thu_de_nghe_va_phan_xet():
    # Thiếu path_norm thì người duyệt không có file để mở.
    assert {"file_id", "class_id", "path_norm", "verdict"} <= set(vs.AUDIT_FIELDS)


# ── Hàng đợi review_queue KHÔNG được tính vào tỉ lệ lỗi τ ───────────────────


def test_hang_doi_khong_tinh_vao_ti_le_loi():
    """review_queue không phải mẫu auto-accept — trộn vào sẽ đo sai đối tượng."""
    mau = [_audited("ok"), {**_audited("wrong_class", "gunshot"), "origin": "sample"}]
    hang_doi = [{**_audited("wrong_class", "siren"), "origin": "queue"}]
    errors, total, _ = vs.error_rate(mau + hang_doi)
    assert total == 2      # chỉ 2 dòng sample, không tính dòng queue
    assert errors == 1


def test_dong_khong_co_origin_duoc_coi_la_sample():
    """Tương thích ngược: 245 dòng cũ trước khi có cột `origin` phải vẫn tính được."""
    cu = _audited("ok")
    cu.pop("origin", None)
    errors, total, _ = vs.error_rate([cu])
    assert total == 1


# ── cmd_queue: gộp review_queue.csv, không đè mẫu đã có ─────────────────────


def test_gop_hang_doi_khong_dung_lai_dong_da_co(tmp_path, monkeypatch):
    monkeypatch.setattr(vs, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(vs, "AUDIT_PATH", tmp_path / "screen_audit.csv")
    monkeypatch.setattr(vs, "QUEUE_PATH", tmp_path / "review_queue.csv")

    import csv
    with vs.AUDIT_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=vs.AUDIT_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerow({**_audited("ok", "siren"), "file_id": "a"})

    queue_fields = ["priority", "file_id", "class_id", "p_target", "p_confusable",
                    "top_confusable", "onset", "offset", "path_norm"]
    with vs.QUEUE_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=queue_fields)
        w.writeheader()
        w.writerow({"priority": "normal", "file_id": "a", "class_id": "siren",
                    "p_target": "0.5", "p_confusable": "0.1", "top_confusable": "",
                    "onset": "", "offset": "", "path_norm": "x/a.wav"})
        w.writerow({"priority": "normal", "file_id": "b", "class_id": "gunshot",
                    "p_target": "0.5", "p_confusable": "0.1", "top_confusable": "",
                    "onset": "", "offset": "", "path_norm": "x/b.wav"})

    assert vs.cmd_queue() == 0
    rows = vs.read_csv(vs.AUDIT_PATH)
    assert {r["file_id"] for r in rows} == {"a", "b"}      # "a" không nhân đôi
    b = next(r for r in rows if r["file_id"] == "b")
    assert b["origin"] == "queue"
    assert b["verdict"] == ""
    a = next(r for r in rows if r["file_id"] == "a")
    assert a["verdict"] == "ok"                            # verdict cũ của "a" còn nguyên


def test_gop_hang_doi_sap_theo_class_id_khong_giu_mau_cu_o_dau(tmp_path, monkeypatch):
    monkeypatch.setattr(vs, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(vs, "AUDIT_PATH", tmp_path / "screen_audit.csv")
    monkeypatch.setattr(vs, "QUEUE_PATH", tmp_path / "review_queue.csv")

    import csv
    with vs.AUDIT_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=vs.AUDIT_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerow({**_audited("ok", "siren"), "file_id": "z_siren"})   # mẫu cũ, "z" đứng cuối theo file_id

    queue_fields = ["priority", "file_id", "class_id", "p_target", "p_confusable",
                    "top_confusable", "onset", "offset", "path_norm"]
    with vs.QUEUE_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=queue_fields)
        w.writeheader()
        w.writerow({"priority": "normal", "file_id": "a_gunshot", "class_id": "gunshot",
                    "p_target": "0.5", "p_confusable": "0.1", "top_confusable": "",
                    "onset": "", "offset": "", "path_norm": "x/a.wav"})

    assert vs.cmd_queue() == 0
    rows = vs.read_csv(vs.AUDIT_PATH)
    # "gunshot" (lớp mới thêm) phải đứng TRƯỚC "siren" (mẫu cũ) — sắp theo class_id,
    # không phải theo thứ tự thêm vào file.
    assert [r["class_id"] for r in rows] == ["gunshot", "siren"]


def test_chua_co_review_queue_thi_bao_loi(tmp_path, monkeypatch):
    monkeypatch.setattr(vs, "AUDIT_PATH", tmp_path / "screen_audit.csv")
    monkeypatch.setattr(vs, "QUEUE_PATH", tmp_path / "khong_ton_tai.csv")
    assert vs.cmd_queue() == 1


# ── cmd_stage: bỏ qua dòng đã có verdict, không đánh số lại ─────────────────


def test_stage_bo_qua_dong_da_xong(tmp_path, monkeypatch):
    monkeypatch.setattr(vs, "AUDIT_PATH", tmp_path / "screen_audit.csv")

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "b.wav").write_bytes(b"RIFF")

    import csv
    with vs.AUDIT_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=vs.AUDIT_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerow({**_audited("ok", "siren"), "file_id": "a", "path_norm": str(audio_dir / "a.wav")})
        w.writerow({**_audited("", "gunshot"), "file_id": "b", "path_norm": str(audio_dir / "b.wav")})

    monkeypatch.setattr(vs, "REPO_ROOT", tmp_path)
    assert vs.cmd_stage() == 0
    staged = sorted(p.name for p in (tmp_path / "data" / "interim" / "audit_playlist").glob("*.wav"))
    assert staged == ["002.wav"]      # dòng 1 ("a") đã xong nên bị bỏ, số 2 giữ đúng vị trí


def test_stage_all_chep_lai_ca_dong_da_xong(tmp_path, monkeypatch):
    monkeypatch.setattr(vs, "AUDIT_PATH", tmp_path / "screen_audit.csv")
    monkeypatch.setattr(vs, "REPO_ROOT", tmp_path)

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "a.wav").write_bytes(b"RIFF")
    (audio_dir / "b.wav").write_bytes(b"RIFF")

    import csv
    with vs.AUDIT_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=vs.AUDIT_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerow({**_audited("ok", "siren"), "file_id": "a", "path_norm": str(audio_dir / "a.wav")})
        w.writerow({**_audited("", "gunshot"), "file_id": "b", "path_norm": str(audio_dir / "b.wav")})

    assert vs.cmd_stage(include_done=True) == 0
    staged = sorted(p.name for p in (tmp_path / "data" / "interim" / "audit_playlist").glob("*.wav"))
    assert staged == ["001.wav", "002.wav"]      # --all: cả dòng đã xong cũng được chép lại
