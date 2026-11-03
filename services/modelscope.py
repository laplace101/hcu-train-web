"""ModelScope config.json retrieval (plan §6.1, C02, C03).

ModelScope is the primary model source for this product. The rules that
matter are all about not being a blind fetcher:

  * only whitelisted hosts (SSRF protection, plan C03)
  * no redirects to other hosts
  * bounded response size and timeouts
  * a specific revision is fetched and recorded, so the result is reproducible
  * external failure never blocks the "upload config.json" path
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Only these hosts may be fetched from. Anything else is rejected before a
#: socket is opened.
ALLOWED_HOSTS = {
    "modelscope.cn",
    "www.modelscope.cn",
    "modelscope.com",
    "www.modelscope.com",
    "modelers.cn",
    "www.modelers.cn",
}

MAX_RESPONSE_BYTES = 2 * 1024 * 1024
TIMEOUT_SECONDS = 30
USER_AGENT = "dcu-train-web/2.0"

#: Cache of fetched configs, keyed by repo+revision. Keeps repeated lookups
#: fast and reduces load on the upstream service.
_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_CACHE_TTL_SECONDS = 3600


class ModelSourceError(ValueError):
    """Raised when a model reference cannot be turned into a config.json."""


@dataclass
class ResolvedConfig:
    """A config.json plus the provenance the plan requires for traceability."""

    config: dict[str, Any]
    repo_id: str | None = None
    revision: str | None = None
    source: str = "upload"
    config_hash: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "source": self.source,
            "config_hash": self.config_hash,
            "warnings": list(self.warnings),
        }


def hash_config(raw: dict[str, Any]) -> str:
    """Stable hash of a config, used to detect upstream changes."""
    blob = json.dumps(raw, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _is_allowed_host(host: str) -> bool:
    host = host.lower().split(":")[0]
    return host in ALLOWED_HOSTS


def parse_modelscope_reference(text: str) -> tuple[str, str | None]:
    """Return (repo_id, revision) from a ModelScope name or URL.

    Accepts:
      * ``org/model``
      * ``org/model@revision``
      * ``https://modelscope.cn/models/org/model``
      * ``https://modelscope.cn/models/org/model/repo?Revision=xxx``
    """
    raw = (text or "").strip()
    if not raw:
        raise ModelSourceError("请输入 ModelScope 模型名称或链接")

    if raw.startswith("ms:"):
        raw = raw[3:].strip()

    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urllib.parse.urlparse(raw)
        if not _is_allowed_host(parsed.netloc):
            raise ModelSourceError(
                f"不支持的域名 {parsed.netloc}；仅允许 "
                f"{', '.join(sorted(ALLOWED_HOSTS))}"
            )
        segments = [s for s in parsed.path.split("/") if s]
        # /models/<org>/<model>[/repo]
        if "models" in segments:
            idx = segments.index("models")
            parts = segments[idx + 1 : idx + 3]
        else:
            parts = segments[:2]
        if len(parts) < 2:
            raise ModelSourceError(f"无法从链接解析仓库名：{raw}")
        repo = "/".join(parts)
        revision = None
        query = urllib.parse.parse_qs(parsed.query)
        if query.get("Revision"):
            revision = query["Revision"][0]
        return repo, revision

    repo = raw.strip("/")
    revision = None
    if "@" in repo:
        repo, revision = repo.rsplit("@", 1)
    if "/" not in repo:
        raise ModelSourceError(
            f"模型名需要包含「组织/模型」两部分，例如 Qwen/Qwen3-8B；当前为 {repo}"
        )
    if not re.fullmatch(r"[\w.\-]+/[\w.\-]+", repo):
        raise ModelSourceError(f"不是合法的 ModelScope 仓库名：{repo}")
    return repo, revision


def _config_urls(repo: str, revision: str | None) -> list[str]:
    """Candidate config.json URLs, most specific first."""
    urls: list[str] = []
    if revision:
        urls.append(
            f"https://modelscope.cn/models/{repo}/resolve/{revision}/config.json"
        )
        urls.append(
            f"https://modelscope.cn/models/{repo}/repo?Revision={revision}&FilePath=config.json"
        )
    urls.append(f"https://modelscope.cn/models/{repo}/resolve/master/config.json")
    urls.append(
        f"https://modelscope.cn/models/{repo}/repo?Revision=master&FilePath=config.json"
    )
    return urls


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects that leave the whitelist."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        parsed = urllib.parse.urlparse(newurl)
        if not _is_allowed_host(parsed.netloc):
            raise ModelSourceError(f"拒绝跳转到非白名单域名：{parsed.netloc}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _build_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_NoRedirect)


def _request_json(url: str) -> tuple[dict[str, Any], str | None]:
    """Fetch a JSON document, bounded in size and time.

    Returns (document, final_url). Raises ModelSourceError on any failure.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    opener = _build_opener()
    try:
        with opener.open(request, timeout=TIMEOUT_SECONDS) as response:
            content_type = response.headers.get("Content-Type", "")
            payload = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise ModelSourceError(f"访问 ModelScope 失败（HTTP {exc.code}）：{url}") from exc
    except ModelSourceError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ModelSourceError(f"访问 ModelScope 失败：{exc}") from exc

    if len(payload) > MAX_RESPONSE_BYTES:
        raise ModelSourceError("config.json 响应超出大小上限，已拒绝")

    text = payload.decode("utf-8", errors="replace").strip()
    if not text:
        raise ModelSourceError("ModelScope 返回空响应")

    # ModelScope's `repo?FilePath=` endpoint sometimes wraps the document.
    if not text.startswith("{"):
        match = re.search(r'"(?:Data|Content|data|content)"\s*:\s*(\{.*\})\s*\}?\s*$', text, re.S)
        if match:
            text = match.group(1)
    try:
        return json.loads(text), url
    except json.JSONDecodeError as exc:
        raise ModelSourceError(f"ModelScope 返回的内容不是合法 JSON：{exc}") from exc


def fetch_modelscope_config(
    text: str, *, use_cache: bool = True
) -> ResolvedConfig:
    """Resolve a ModelScope reference into a config.json with provenance."""
    repo, revision = parse_modelscope_reference(text)
    cache_key = f"{repo}@{revision or 'master'}"

    if use_cache:
        hit = _CACHE.get(cache_key)
        if hit and (time.time() - hit[0]) < _CACHE_TTL_SECONDS:
            config = hit[1]
            return ResolvedConfig(
                config=config,
                repo_id=repo,
                revision=revision or "master",
                source="modelscope",
                config_hash=hash_config(config),
            )

    last_error: Exception | None = None
    for url in _config_urls(repo, revision):
        try:
            document, final_url = _request_json(url)
        except ModelSourceError as exc:
            last_error = exc
            continue

        config = document
        # Some responses nest the real config under a "config" key.
        if isinstance(config, dict) and "config" in config and isinstance(config["config"], dict):
            config = config["config"]
        if not isinstance(config, dict):
            last_error = ModelSourceError("config.json 顶层不是 JSON 对象")
            continue

        resolved_revision = revision
        if not resolved_revision:
            match = re.search(r"/resolve/([^/]+)/", final_url)
            if match:
                resolved_revision = match.group(1)

        _CACHE[cache_key] = (time.time(), config)
        return ResolvedConfig(
            config=config,
            repo_id=repo,
            revision=resolved_revision or "master",
            source="modelscope",
            config_hash=hash_config(config),
        )

    raise ModelSourceError(
        f"未能从 {repo} 获取 config.json（{last_error}）。"
        "可改用「上传 config.json」方式。"
    )


def parse_uploaded_json(text: str) -> ResolvedConfig:
    """Parse a pasted or uploaded config.json."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ModelSourceError(f"config.json 不是合法 JSON：{exc}") from exc
    if not isinstance(data, dict):
        raise ModelSourceError("config.json 顶层必须是 JSON 对象")
    return ResolvedConfig(
        config=data,
        source="upload",
        config_hash=hash_config(data),
    )


def store_config(resolved: ResolvedConfig, dest_dir: Path, slug_hint: str = "uploaded") -> Path:
    """Persist a config to disk and return the written path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.\-]+", "-", slug_hint).strip("-") or "model"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = dest_dir / f"{safe}-{stamp}"
    target.mkdir(parents=True, exist_ok=True)
    path = target / "config.json"
    path.write_text(
        json.dumps(resolved.config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path
