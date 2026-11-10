"""Checks for the measured-benchmark integration.

Verifies the read-only guarantee and the comparison logic.

    /root/.local/share/uv/python/cpython-3.14.7-linux-x86_64-gnu/bin/python3.14
    tests/test_measured.py
"""

from __future__ import annotations

import hashlib
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import app  # noqa: E402
from repositories import measured as measured_repo  # noqa: E402

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def db_checksum() -> str:
    path = measured_repo.database_path()
    if path is None:
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    before = db_checksum()

    print("=== ① 数据库可读 ===")
    check("找到数据库", measured_repo.available(), str(measured_repo.database_path()))
    info = measured_repo.database_info()
    check("标记为只读", info.get("read_only") is True)
    check("有训练数据", (info.get("total_runs") or 0) > 0, f"{info.get('total_runs')} 条")
    check("区分成功与 OOM",
          (info.get("ok_runs") or 0) > 0 and (info.get("oom_runs") or 0) > 0,
          f"OK {info.get('ok_runs')} / OOM {info.get('oom_runs')}")

    print("\n=== ② 写入被拒绝 ===")
    path = measured_repo.database_path()
    if path is not None:
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            conn.execute("UPDATE train_bench SET note='TAMPERED' WHERE id=1")
            conn.commit()
            check("只读连接拒绝写入", False, "写入竟然成功了")
        except sqlite3.OperationalError as exc:
            check("只读连接拒绝写入", "readonly" in str(exc).lower(), str(exc)[:60])
        except Exception as exc:  # noqa: BLE001
            check("只读连接拒绝写入", False, f"{type(exc).__name__}: {exc}")

    print("\n=== ③ 模型名与目录 slug 对应 ===")
    summary = measured_repo.summarise_for_slug("qwen3-5-2b")
    check("qwen3-5-2b 匹配到实测行", (summary.get("count") or 0) > 0,
          f"{summary.get('count')} 条, 匹配名 {summary.get('model_names')}")
    check("给出最佳吞吐", summary.get("best_tokens_per_s") is not None,
          str(summary.get("best_tokens_per_s")))
    check("给出显存区间",
          summary.get("peak_mem_gib_min") is not None,
          f"{summary.get('peak_mem_gib_min')}~{summary.get('peak_mem_gib_max')} GiB")
    check("记录卡数分布", bool(summary.get("gpu_counts")), str(summary.get("gpu_counts")))

    missing = measured_repo.summarise_for_slug("deepseek-v3")
    check("无数据的模型给出原因", not missing.get("rows") and bool(missing.get("reason")),
          missing.get("reason", ""))

    print("\n=== ④ 实测模型清单 ===")
    models = measured_repo.list_measured_models()
    check("列出模型", len(models) > 0, f"{len(models)} 个")
    check("含成功计数", all("ok_runs" in m for m in models))

    print("\n=== ⑤ API 端点 ===")
    client = app.test_client()

    r = client.get("/api/v1/measured/info")
    body = r.get_json()
    check("measured/info 200", r.status_code == 200 and body["ok"])
    check("声明只读", body.get("read_only") is True)

    r = client.get("/api/v1/measured/models")
    body = r.get_json()
    check("measured/models 200", r.status_code == 200 and body["ok"])
    check("有模型列表", body["count"] > 0, f"{body['count']} 个")

    r = client.get("/api/v1/measured/summary?slug=qwen3-5-2b")
    body = r.get_json()
    check("measured/summary 200", r.status_code == 200 and body["ok"])
    check("返回实测行", body["count"] > 0, f"{body['count']} 条")

    r = client.get("/api/v1/measured?slug=qwen3-5-2b&limit=5")
    body = r.get_json()
    check("measured 查询 200", r.status_code == 200 and body["ok"])
    check("limit 生效", body["count"] <= 5, str(body["count"]))
    check("行含吞吐字段",
          all("tokens_per_s" in row for row in body["rows"]),
          str(body["rows"][0].get("tokens_per_s")) if body["rows"] else "")

    r = client.get("/api/v1/measured?gpu_count=8&status=OK")
    body = r.get_json()
    check("按卡数过滤", r.status_code == 200
          and all(row["gpu_count"] == 8 for row in body["rows"] if row.get("gpu_count")))

    print("\n=== ⑥ 估算附带实测 ===")
    r = client.post("/api/v1/training/estimates", json={
        "mode": "sft_full",
        "hardware_id": "bw1000",
        "model_slug": "qwen3-5-2b",
        "model_config_path": str(
            PROJECT_ROOT / "catalog" / "models" / "qwen3-5-2b" / "config.json"),
        "parallel": {"tp_size": 1, "pp_size": 1, "cp_size": 1, "ep_size": 1,
                     "num_gpus": 8, "micro_batch_size": 4,
                     "global_batch_size": 256, "seq_length": 2048},
    })
    body = r.get_json()
    check("estimates 200", r.status_code == 200 and body["ok"])
    m = body.get("measured") or {}
    check("附带 measured", m.get("available") is True)
    check("匹配到模型", m.get("matched") is True, str(m.get("slug")))
    check("给出估算与实测吞吐",
          m.get("estimate_tgs") is not None and m.get("measured_tgs") is not None,
          f"est {m.get('estimate_tgs')} / meas {m.get('measured_tgs')}")
    check("说明可比性", bool(m.get("comparability_note")))

    # The full-parameter estimate is compared against full-parameter
    # measurements. The database holds one such run for this model, and it
    # ran at a different batch size, so the comparison is *not* like for
    # like and the deviation is withheld rather than reported as if it
    # were the estimator's error.
    check("全参估算优先匹配全参实测", m.get("measured_is_lora") is False,
          f"measured_is_lora={m.get('measured_is_lora')}")
    check("同口径时标记为可比", m.get("comparable") is True,
          f"comparable={m.get('comparable')}")
    check("标注是否同配置", m.get("like_for_like") in (True, False),
          f"like_for_like={m.get('like_for_like')}")
    if m.get("like_for_like") is False:
        check("配置不同时不给偏差", m.get("tgs_deviation_percent") is None,
              f"{m.get('tgs_deviation_percent')}")
        check("说明配置差异", bool(m.get("match_differences")),
              "；".join(m.get("match_differences") or []))
    else:
        check("同配置时给出偏差", m.get("tgs_deviation_percent") is not None,
              f"{m.get('tgs_deviation_percent')}%")

    print("\n=== ⑦ 旧接口也带实测 ===")
    r = client.post("/api/simulate", json={
        "task": "full", "hardware": "bw1000", "operator_profile": "bw1000",
        "model": "qwen3-5-2b",
        "parallel": {"tp_size": 1, "pp_size": 1, "cp_size": 1, "ep_size": 1,
                     "num_gpus": 8, "micro_batch_size": 4,
                     "global_batch_size": 256, "seq_length": 2048},
    })
    body = r.get_json()
    check("simulate 200", r.status_code == 200 and body["ok"])
    m = (body.get("result") or {}).get("measured") or {}
    check("旧接口也带 measured", m.get("available") is True and m.get("matched") is True,
          str(m.get("slug")))

    print("\n=== ⑧ 数据库未被修改 ===")
    after = db_checksum()
    check("校验和不变", before == after, f"{before[:16]}…")
    if path is not None:
        # The data must still be intact and readable.
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        count = conn.execute("SELECT COUNT(*) FROM train_bench").fetchone()[0]
        tampered = conn.execute(
            "SELECT COUNT(*) FROM train_bench WHERE note='TAMPERED'").fetchone()[0]
        conn.close()
        check("行数未变", count == (info.get("total_runs") or 0), str(count))
        check("无篡改痕迹", tampered == 0, str(tampered))

    print("\n" + "=" * 46)
    if failures:
        print(f"FAILED {len(failures)}: {', '.join(failures)}")
        return 1
    print("ALL MEASURED-DATA CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
