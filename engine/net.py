"""Network access helpers.

Windows stores its proxy in the WinINET registry, but Python's urllib/requests
only read the HTTP_PROXY / HTTPS_PROXY environment variables. On this machine
the system proxy is `127.0.0.1:10808`, so without this shim every HuggingFace
request hangs until timeout. We mirror the WinINET proxy into the environment
once, before any HTTP client is constructed.
"""

from __future__ import annotations

import os

_APPLIED = False


def _wininet_proxy() -> str | None:
    if os.name != "nt":
        return None
    try:
        import winreg
    except ImportError:  # pragma: no cover - non-Windows
        return None

    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
        )
    except OSError:
        return None

    try:
        enable, _ = winreg.QueryValueEx(key, "ProxyEnable")
        if not enable:
            return None
        server, _ = winreg.QueryValueEx(key, "ProxyServer")
    except OSError:
        return None
    finally:
        winreg.CloseKey(key)

    server = str(server).strip()
    if not server:
        return None
    if "=" in server:  # per-protocol form, e.g. "https=host:port;http=host:port"
        parsed = {}
        for item in server.split(";"):
            if "=" in item:
                proto, _, value = item.partition("=")
                parsed[proto.strip().lower()] = value.strip()
        server = parsed.get("https") or parsed.get("http") or ""
    if not server:
        return None
    if "://" not in server:
        server = f"http://{server}"
    return server


def apply_proxy() -> str | None:
    """Populate HTTP(S)_PROXY from the Windows system proxy if unset."""

    global _APPLIED
    if _APPLIED:
        return os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")

    _APPLIED = True
    if os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY"):
        return os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")

    proxy = _wininet_proxy()
    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy
        no_proxy = os.environ.get("NO_PROXY")
        os.environ["NO_PROXY"] = (
            f"{no_proxy},localhost,127.0.0.1" if no_proxy else "localhost,127.0.0.1"
        )
    return proxy