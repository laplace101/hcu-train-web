"""TCP reachability probe for the BW-class hosts in ~/.ssh/config."""
import socket
import sys

CANDIDATES = [
    ("BW1000",           "10.17.26.195", 22),
    ("BW1100",           "10.17.26.197", 22),
    ("tongxinBW1000",    "10.17.30.130", 22),
    ("megatron",         "10.17.30.130", 10088),
    ("hanghai-jifang",   "10.17.29.193", 22),
    ("BW100",            "10.17.28.100", 22),
    ("hanghang-BW1000",  "10.20.100.17", 22),
    ("zhanbei1",         "10.8.145.254", 22),
    ("zhanbei2",         "10.8.145.253", 22),
    ("A800",             "10.17.26.244", 22),
]


def probe(name, host, port, timeout=2.5):
    sock = socket.socket()
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        return True, "OPEN"
    except socket.timeout:
        return False, "timeout (filtered/unreachable)"
    except OSError as exc:
        return False, f"{exc.__class__.__name__}: {exc.strerror or exc}"
    finally:
        sock.close()


print(f"{'alias':20s} {'host':16s} {'port':6s} result")
print("-" * 74)
for name, host, port in CANDIDATES:
    ok, detail = probe(name, host, port)
    print(f"{name:20s} {host:16s} {port:<6d} {'REACHABLE' if ok else 'unreachable'}  ({detail})")