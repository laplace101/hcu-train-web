"""Resolve a user-supplied model reference to a HuggingFace config.json.

Accepted inputs (the UI offers three radio buttons):
  * preset          - a slug already vendored under catalog/models
  * hf              - a repo id, a huggingface.co URL, or a /resolve/<rev>/ URL
  * upload          - an uploaded or pasted config.json
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from . import net, paths

HF_HOSTS = {"huggingface.co", "www.huggingface.co", "hf.co"}
RAW_MARKERS = ("/raw/", "/resolve/", "/blob/")
_USER_AGENT = "hcu-train-web/1.0"

# HF api models: returns the commit sha a config.json should be read from.
_API_URL = "https://huggingface.co/api/models/{repo}"


class ModelSourceError(ValueError):
    """Raised when a model reference cannot be turned into a config.json."""


def _request_json(url: str, timeout: int = 45) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_hf_reference(text: str) -> tuple[str, str | None]:
    """Return (repo_id, revision) from a repo id or any common HF URL form."""

    raw = (text or "").strip()
    if not raw:
        raise ModelSourceError("请输入 HuggingFace 模型名称或链接")

    if raw.startswith("hf:"):
        raw = raw[3:].strip()

    candidate = raw
    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urllib.parse.urlparse(raw)
        host = parsed.netloc.lower()
        if host not in HF_HOSTS:
            raise ModelSourceError(
                f"仅支持 huggingface.co 链接，当前为 {host}；请改用「Qwen/Qwen3-8B」这样的仓库名"
            )
        segments = [s for s in parsed.path.split("/") if s]
        if len(segments) < 2:
            raise ModelSourceError(f"无法从链接中解析仓库名: {raw}")
        repo = "/".join(segments[:2])
        revision = None
        for marker in RAW_MARKERS:
            if marker in parsed.path:
                tail = parsed.path.split(marker, 1)[1].strip("/")
                if tail:
                    parts = tail.split("/")
                    if parts and parts[0] and not parts[0].startswith("main"):
                        revision = parts[0]
                    break
        return repo, revision

    # Bare "owner/name", optionally with a revision suffix.
    repo = candidate.strip("/")
    revision = "main"
    if "@" in repo:
        repo, revision = repo.rsplit("@", 1)
    if "/" not in repo:
        raise ModelSourceError(
            f"模型名需要包含「组织/模型」两部分，例如 Qwen/Qwen3-8B；当前为 {repo}"
        )
    if not re.fullmatch(r"[\w.\-]+/[\w.\-]+", repo):
        raise ModelSourceError(f"不是合法的 HuggingFace 仓库名: {repo}")
    return repo, revision


def fetch_hf_config(text: str) -> dict[str, Any]:
    """Download the config.json for a repo id / HF link."""

    net.apply_proxy()
    repo, revision = parse_hf_reference(text)

    api_url = _API_URL.format(repo=repo)
    try:
        meta = _request_json(api_url)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            # HuggingFace answers 401 for gated repos *and* for repos that do
            # not exist, so we cannot tell the two apart anonymously.
            raise ModelSourceError(
                f"{repo} 无法匿名访问：可能是私有仓库、Gated 需授权，或仓库名不存在"
                "（HF 对后三者都返回 401）。请确认拼写，或改用「上传 config.json」方式。"
            ) from exc
        if exc.code == 404:
            raise ModelSourceError(f"仓库不存在或没有 config.json: {repo}") from exc
        raise ModelSourceError(f"访问 HuggingFace 失败（HTTP {exc.code}）: {repo}") from exc
    except Exception as exc:  # noqa: BLE001
        raise ModelSourceError(f"访问 HuggingFace 失败: {exc}") from exc

    sha = meta.get("sha") or revision
    candidates = [
        f"https://huggingface.co/{repo}/resolve/{sha}/config.json",
        f"https://huggingface.co/{repo}/raw/{sha}/config.json",
        f"https://huggingface.co/{repo}/resolve/main/config.json",
    ]
    last_error: Exception | None = None
    for url in candidates:
        try:
            return _request_json(url)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in (403, 404):
                break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            break

    if isinstance(last_error, urllib.error.HTTPError) and last_error.code in (401, 403):
        raise ModelSourceError(
            f"{repo} 需要授权访问，无法下载 config.json；请使用「上传 config.json」"
        )
    raise ModelSourceError(f"未能在 {repo} 中找到 config.json（{last_error}）")


def store_config(raw: dict[str, Any], slug_hint: str = "uploaded") -> Path:
    """Persist a config dict into the upload area and return its path."""

    paths.ensure_dirs()
    safe = re.sub(r"[^A-Za-z0-9_.\-]+", "-", slug_hint).strip("-") or "model"
    stamp = __import__("time").strftime("%Y%m%d-%H%M%S")
    target_dir = paths.UPLOAD_DIR / f"{safe}-{stamp}"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "config.json"
    path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def parse_uploaded_json(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ModelSourceError(f"config.json 不是合法 JSON：{exc}") from exc
    if not isinstance(data, dict):
        raise ModelSourceError("config.json 顶层必须是 JSON 对象")
    return data