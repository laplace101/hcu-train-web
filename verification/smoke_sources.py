"""Smoke-test the model-source endpoints (HF link / gated / upload / paste)."""
import json
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:7788"


def post(path, payload=None, files=None):
    if files:
        boundary = "----hcuwebtest"
        parts = []
        for name, value in payload.items():
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())
        for name, (filename, content) in files.items():
            parts.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
                f"Content-Type: application/json\r\n\r\n".encode() + content + b"\r\n"
            )
        parts.append(f"--{boundary}--\r\n".encode())
        body = b"".join(parts)
        content_type = f"multipart/form-data; boundary={boundary}"
    else:
        body = json.dumps(payload).encode()
        content_type = "application/json"

    req = urllib.request.Request(BASE + path, data=body, headers={"Content-Type": content_type})
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


print("=== HF link ===")
status, data = post("/api/resolve-model", {"source": "hf", "hf_ref": "https://huggingface.co/Qwen/Qwen3-32B"})
print(status, data.get("ok"), data.get("label"), (data.get("summary") or {}).get("adapter"))

print("\n=== HF bare repo id ===")
status, data = post("/api/resolve-model", {"source": "hf", "hf_ref": "deepseek-ai/DeepSeek-V3"})
print(status, data.get("ok"), data.get("label"), (data.get("summary") or {}).get("adapter"))

print("\n=== HF gated repo (expect friendly error) ===")
status, data = post("/api/resolve-model", {"source": "hf", "hf_ref": "meta-llama/Llama-3.1-8B"})
print(status, data.get("ok"), data.get("error"))

print("\n=== HF nonexistent repo ===")
status, data = post("/api/resolve-model", {"source": "hf", "hf_ref": "Qwen/does-not-exist-xyz"})
print(status, data.get("ok"), data.get("error"))

print("\n=== upload config.json ===")
cfg = json.dumps({
    "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 4096,
    "num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8,
    "intermediate_size": 12288, "vocab_size": 151936, "head_dim": 128,
}).encode()
status, data = post("/api/resolve-model", payload={"source": "upload"}, files={"file": ("my_model.json", cfg)})
print(status, data.get("ok"), data.get("label"), (data.get("summary") or {}).get("adapter"))
print("   model_ref:", data.get("model_ref"))

print("\n=== paste config.json ===")
status, data = post("/api/resolve-model", {"source": "paste", "config_text": cfg.decode()})
print(status, data.get("ok"), data.get("label"), (data.get("summary") or {}).get("adapter"))

print("\n=== paste invalid json ===")
status, data = post("/api/resolve-model", {"source": "paste", "config_text": "{not json"})
print(status, data.get("ok"), data.get("error"))