"""Training image and software-stack catalogue (plan G01, G02).

Source of truth: the DCU image-hub snapshot shipped with the toolkit skills
(`dcu-image-hub/data/sourcefind_dcu_images.json`), which lists every public
Harbor repository with its tags and pull URLs.

Two rules from the plan are enforced here rather than in the UI:

  * G02 — extraction codes (提取码) are secrets. They are stored hashed and
    never returned in plain text; the list shows only a masked hint and a
    `has_code` flag.
  * G01 — a tag is only marked *usable* when it names a framework the
    training templates need (Megatron / DeepSpeed / Torch / vLLM / SGLang
    …). Anything else stays in the catalogue but is not offered as a
    training image.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from pathlib import Path
from typing import Any

APP_ROOT = Path(__file__).resolve().parents[1]

#: Where the image-hub snapshot lives. Overridable for deployments that
#: refresh the catalogue.
IMAGE_INDEX_PATHS = [
    Path(os.environ.get("DCU_IMAGE_INDEX", "")) if os.environ.get("DCU_IMAGE_INDEX") else None,
    APP_ROOT / "data" / "sourcefind_dcu_images.json",
    Path("/root/hygon/dcu-toolkit-skills-master/env/dcu-image-hub/data/sourcefind_dcu_images.json"),
]

#: Frameworks the training templates need, mapped to the tag regex that
#: identifies them. The order matters: the first match wins.
TRAINING_FRAMEWORKS: list[tuple[str, str]] = [
    ("megatron", r"megatron|mgcore|mcore"),
    ("deepspeed", r"deepspeed"),
    ("msit", r"msit"),
    ("verl", r"verl"),
    ("llamafactory", r"llama.?factory"),
    ("swift", r"ms-?swift|swift"),
    ("xtuner", r"xtuner"),
    ("torchtune", r"torchtune"),
    ("transformers", r"transformers"),
    ("pytorch", r"pytorch|torch"),
    ("vllm", r"vllm"),
    ("sglang", r"sglang"),
]

#: DTK version, extracted from the tag.
_DTK_RE = re.compile(r"dtk(\d+(?:\.\d+)*)")
_PY_RE = re.compile(r"py(\d+(?:\.\d+)*)")
_UBUNTU_RE = re.compile(r"ubuntu(\d+(?:\.\d+)*)")
_HCU_RE = re.compile(r"\b(z100|k100|bw1000|k100i|bw500sm|bw1100)\b", re.I)

#: Salt for the extraction-code hash. Kept out of the repository.
_CODE_SALT = os.environ.get("DCU_IMAGE_CODE_SALT", "hcu-train-web")


def _load_index() -> dict[str, Any]:
    for candidate in IMAGE_INDEX_PATHS:
        if candidate and candidate.exists():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict) and "images" in data:
                return data
    return {"images": [], "source": None, "fetched_at": None}


_INDEX = _load_index()


def source_info() -> dict[str, Any]:
    """Where the catalogue came from and when it was fetched."""
    for candidate in IMAGE_INDEX_PATHS:
        if candidate and candidate.exists():
            return {
                "path": str(candidate),
                "fetched_at": _INDEX.get("fetched_at"),
                "source": _INDEX.get("source"),
            }
    return {"path": None, "fetched_at": None, "source": None}


def _detect_framework(tag_name: str) -> str | None:
    lowered = tag_name.lower()
    for framework, pattern in TRAINING_FRAMEWORKS:
        if re.search(pattern, lowered):
            return framework
    return None


def _parse_tag(tag: dict[str, Any]) -> dict[str, Any]:
    name = tag.get("name", "")
    dtk = _DTK_RE.search(name)
    python = _PY_RE.search(name)
    ubuntu = _UBUNTU_RE.search(name)
    hcu = _HCU_RE.search(name)
    return {
        "name": name,
        "size": tag.get("size"),
        "pull_url": tag.get("pull_url"),
        "push_time": tag.get("push_time"),
        "release_type": tag.get("release_type"),
        "framework": _detect_framework(name),
        "dtk_version": f"DTK {dtk.group(1)}" if dtk else None,
        "python_version": f"py{python.group(1)}" if python else None,
        "os_version": f"ubuntu{ubuntu.group(1)}" if ubuntu else None,
        "accelerator": hcu.group(1).upper() if hcu else None,
        "usable_for_training": _detect_framework(name) is not None,
    }


def list_images(*, framework: str | None = None, keyword: str | None = None) -> list[dict[str, Any]]:
    """List training images, newest first.

    Only images with at least one training-framework tag are returned; a
    repository with no usable tag would only mislead the user.
    """
    out: list[dict[str, Any]] = []
    keyword_l = (keyword or "").lower()

    for image in _INDEX.get("images", []):
        tags = [_parse_tag(t) for t in image.get("tags", [])]
        usable = [t for t in tags if t["usable_for_training"]]
        if framework:
            usable = [t for t in usable if t["framework"] == framework]
        if not usable:
            continue
        if keyword_l:
            haystack = f"{image.get('repo', '')} {image.get('abstract', '')} " + " ".join(
                t["name"] for t in usable
            )
            if keyword_l not in haystack.lower():
                continue

        out.append(
            {
                "repo": image.get("repo"),
                "project": image.get("project"),
                "accelerator": image.get("accelerator"),
                "abstract": image.get("abstract"),
                "detail_url": image.get("detail_url"),
                "updated": image.get("updated"),
                "tags": sorted(usable, key=lambda t: t.get("push_time") or "", reverse=True),
            }
        )

    out.sort(key=lambda i: i.get("updated") or "", reverse=True)
    return out


def list_frameworks() -> list[dict[str, Any]]:
    """Which frameworks have images, with counts."""
    counts: dict[str, int] = {}
    for image in list_images():
        for tag in image["tags"]:
            counts[tag["framework"]] = counts.get(tag["framework"], 0) + 1
    return [
        {"framework": name, "tag_count": count}
        for name, count in sorted(counts.items(), key=lambda kv: -kv[1])
    ]


def get_image(repo: str) -> dict[str, Any] | None:
    for image in list_images():
        if image["repo"] == repo:
            return image
    return None


# ---------------------------------------------------------------------------
# Software stacks (plan G01)
# ---------------------------------------------------------------------------

#: A software stack is a (framework, DTK, Python, OS) combination that a
#: training template has actually been verified against. Images are matched
#: against these tuples.
SOFTWARE_STACKS: list[dict[str, Any]] = [
    {"id": "megatron-dtk26-py310", "framework": "megatron", "dtk": "26.0",
     "python": "3.10", "os": "ubuntu22.04", "verified": True,
     "note": "Megatron-LM 全参预训练与 SFT 模板"},
    {"id": "deepspeed-dtk26-py310", "framework": "deepspeed", "dtk": "26.0",
     "python": "3.10", "os": "ubuntu22.04", "verified": True,
     "note": "DeepSpeed ZeRO-2/3 全参微调模板"},
    {"id": "verl-dtk26-py310", "framework": "verl", "dtk": "26.0",
     "python": "3.10", "os": "ubuntu22.04", "verified": False,
     "note": "RL 训练（PPO/GRPO）模板，尚未在目标环境验证"},
    {"id": "llamafactory-dtk25-py310", "framework": "llamafactory", "dtk": "25.0",
     "python": "3.10", "os": "ubuntu22.04", "verified": True,
     "note": "LLaMA-Factory LoRA / DPO 模板"},
]


def list_software_stacks() -> list[dict[str, Any]]:
    return list(SOFTWARE_STACKS)


def match_stacks(framework: str) -> list[dict[str, Any]]:
    return [s for s in SOFTWARE_STACKS if s["framework"] == framework]


# ---------------------------------------------------------------------------
# Extraction codes (plan G02)
# ---------------------------------------------------------------------------


def hash_code(code: str) -> str:
    """Hash an extraction code for storage."""
    return hmac.new(
        _CODE_SALT.encode("utf-8"), code.strip().encode("utf-8"), hashlib.sha256
    ).hexdigest()


def store_code(repo: str, code: str) -> dict[str, Any]:
    """Store an extraction code for a repository.

    The plaintext is never kept; only the HMAC hash is, so the value can be
    verified later but not read back.
    """
    path = _code_store_path()
    store = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    store[repo] = {"hash": hash_code(code), "updated": _today()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"repo": repo, "has_code": True, "updated": store[repo]["updated"]}


def verify_code(repo: str, code: str) -> bool:
    path = _code_store_path()
    if not path.exists():
        return False
    store = json.loads(path.read_text(encoding="utf-8"))
    entry = store.get(repo)
    if not entry:
        return False
    return hmac.compare_digest(entry["hash"], hash_code(code))


def list_codes() -> list[dict[str, Any]]:
    """List stored codes in masked form — never the plaintext."""
    path = _code_store_path()
    if not path.exists():
        return []
    store = json.loads(path.read_text(encoding="utf-8"))
    return [
        {"repo": repo, "has_code": True, "code_hint": "••••", "updated": entry.get("updated")}
        for repo, entry in sorted(store.items())
    ]


def _code_store_path() -> Path:
    return APP_ROOT / "runtime" / "image_codes.json"


def _today() -> str:
    import datetime

    return datetime.date.today().isoformat()
