"""Hardware and software stack catalogue (plan A02, §7.1 `hardware_profiles`).

Design rule that follows from the simulator's actual capabilities, not from
wishful thinking:

  The simulator hardcodes `gpu_model=GPUType.BW1000` in its communication
  estimator, and ships measured bandwidth tables only for `bw1000_*` and
  `bw500SM_*`. A card therefore qualifies as "usable" only when it has
  **both** a measured bandwidth table **and** the hardware parameters the
  simulator expects. Anything else would produce numbers that look precise
  but are invented — exactly what the plan forbids ("不返回伪精确数字").

Cards that exist but are not yet modelled are still listed, with an explicit
reason, so the UI can say "not supported" instead of silently omitting them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HardwareProfile:
    """One DCU model."""

    id: str
    name: str
    #: Usable for estimates. When False the UI must refuse to estimate.
    supported: bool
    unsupported_reason: str = ""
    hbm_gib: int = 64
    fp16_tflops: float = 480.0
    fp8_tflops: float = 960.0
    gpus_per_node: int = 8
    #: The simulator's measured bandwidth tables. Empty means "no measured
    #: table exists" and the card cannot be estimated.
    bandwidth_tables: tuple[str, ...] = ()
    #: Bundled operator profile ids from the simulator (affects compute time).
    operator_profiles: tuple[str, ...] = ()
    gemm_efficiency: float = 0.40
    non_gemm_efficiency: float = 0.08
    optimizer_efficiency: float = 0.04
    p2p_intra_efficiency: float = 0.80
    collective_intra_efficiency: float = 0.70
    collective_inter_efficiency: float = 0.80
    #: Free-form notes shown in the UI.
    notes: str = ""
    aliases: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "supported": self.supported,
            "unsupported_reason": self.unsupported_reason,
            "hbm_gib": self.hbm_gib,
            "fp16_tflops": self.fp16_tflops,
            "fp8_tflops": self.fp8_tflops,
            "gpus_per_node": self.gpus_per_node,
            "bandwidth_tables": list(self.bandwidth_tables),
            "operator_profiles": list(self.operator_profiles),
            "notes": self.notes,
            "aliases": list(self.aliases),
        }


#: Only BW1000 qualifies today: it has both measured bandwidth tables
#: (`bw1000_1.txt` / `bw1000_2.txt`) and a validated operator profile.
HARDWARE_PROFILES: dict[str, HardwareProfile] = {
    "bw1000": HardwareProfile(
        id="bw1000",
        name="海光 DCU BW1000（Z100 / K100）",
        supported=True,
        hbm_gib=64,
        fp16_tflops=480.0,
        fp8_tflops=960.0,
        gpus_per_node=8,
        bandwidth_tables=("bw1000_1.txt", "bw1000_2.txt"),
        operator_profiles=("bw1000_torch271_dtk2604",),
        notes=(
            "唯一同时具备实测带宽表与算子 Profile 的卡型。"
            "HSL 卡间互联 310GB/s，不受 ACS/IOMMU 与虚拟化影响。"
        ),
        aliases=("bw1000", "z100", "k100"),
    ),
    "bw1100": HardwareProfile(
        id="bw1100",
        name="海光 DCU BW1100",
        supported=False,
        unsupported_reason=(
            "仓库打包了 bw1100 算子 Profile（可用于计算时间），"
            "但通信估算器把 gpu_model 写死为 BW1000，且没有 bw1100 实测带宽表。"
            "缺少带宽表时给出带宽数字只能是编造，因此暂不提供估算。"
        ),
        hbm_gib=64,
        fp16_tflops=480.0,
        fp8_tflops=960.0,
        gpus_per_node=8,
        bandwidth_tables=(),
        operator_profiles=("bw1100_torch271_dtk2604",),
        notes="最新一代 HSL 互联机型，算子 Profile 已就绪，等待带宽表补齐。",
        aliases=("bw1100", "bw151"),
    ),
    "bw500sm": HardwareProfile(
        id="bw500sm",
        name="海光 DCU BW500SM",
        supported=False,
        unsupported_reason=(
            "仓库带有 bw500SM 实测带宽表，但缺少对应的算子 Profile 与完整的硬件参数，"
            "计算时间只能退化为理论值，与 BW1000 的口径不可比。"
        ),
        hbm_gib=32,
        fp16_tflops=220.0,
        fp8_tflops=440.0,
        gpus_per_node=8,
        bandwidth_tables=("bw500SM_1.txt", "bw500SM_2.txt"),
        operator_profiles=(),
        notes="入门级训练卡。",
        aliases=("bw500sm", "bw500"),
    ),
    "k100ai": HardwareProfile(
        id="k100ai",
        name="海光 DCU K100-AI",
        supported=False,
        unsupported_reason=(
            "K100-AI 与 BW1000 共用驱动栈但缺少独立带宽表与算子 Profile；"
            "开启 ACS/IOMMU 时 D2D 带宽还会额外受限，无法给出可复现的估算。"
        ),
        hbm_gib=48,
        fp16_tflops=312.0,
        fp8_tflops=624.0,
        gpus_per_node=8,
        bandwidth_tables=(),
        operator_profiles=(),
        notes="开 ACS/IOMMU 时 D2D 带宽受限（NBIO 瓶颈），虚拟化场景需注意。",
        aliases=("k100ai", "k100-ai", "k100_ai"),
    ),
}


def _normalise(text: str) -> str:
    return text.strip().lower().replace("-", "").replace("_", "").replace(" ", "")


_ALIAS_INDEX: dict[str, str] = {}
for _profile in HARDWARE_PROFILES.values():
    _ALIAS_INDEX[_normalise(_profile.id)] = _profile.id
    for _alias in _profile.aliases:
        _ALIAS_INDEX[_normalise(_alias)] = _profile.id


def resolve_hardware(key: str | None) -> HardwareProfile | None:
    """Resolve a card id or alias to its profile."""
    if not key:
        return None
    canonical = _ALIAS_INDEX.get(_normalise(key))
    return HARDWARE_PROFILES.get(canonical) if canonical else None


def supported_hardware() -> list[HardwareProfile]:
    return [p for p in HARDWARE_PROFILES.values() if p.supported]


def describe_hardware() -> list[dict[str, Any]]:
    """Full catalogue for the API, including unsupported cards with reasons."""
    return [p.to_dict() for p in HARDWARE_PROFILES.values()]


#: Software stacks the simulator can be configured for (plan A02).
@dataclass(frozen=True)
class SoftwareStack:
    id: str
    name: str
    driver: str
    dtk: str
    torch: str
    python: str
    os_name: str
    accelerator: str
    supported: bool = True
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "driver": self.driver,
            "dtk": self.dtk,
            "torch": self.torch,
            "python": self.python,
            "os": self.os_name,
            "accelerator": self.accelerator,
            "supported": self.supported,
            "note": self.note,
        }


SOFTWARE_STACKS: tuple[SoftwareStack, ...] = (
    SoftwareStack(
        id="bw1000-torch271-dtk2604",
        name="BW1000 · DTK 26.04 · PyTorch 2.7.1 · Python 3.10",
        driver="rock-6.3.30+",
        dtk="26.04",
        torch="2.7.1",
        python="3.10",
        os_name="ubuntu22.04",
        accelerator="bw1000",
        note="仓库内置算子 Profile 对应的软件栈，估算默认使用。",
    ),
    SoftwareStack(
        id="bw1000-torch271-dtk2604-te210",
        name="BW1000 · DTK 26.04 · PyTorch 2.7.1 · TE 2.10 · Python 3.10",
        driver="rock-6.3.30+",
        dtk="26.04",
        torch="2.7.1",
        python="3.10",
        os_name="ubuntu22.04",
        accelerator="bw1000",
        note="需要 TransformerEngine 2.10 的 MLA / MTP 结构时使用。",
    ),
    SoftwareStack(
        id="bw1100-torch271-dtk2604",
        name="BW1100 · DTK 26.04 · PyTorch 2.7.1 · Python 3.10",
        driver="rock-6.3.30+",
        dtk="26.04",
        torch="2.7.1",
        python="3.10",
        os_name="ubuntu22.04",
        accelerator="bw1100",
        supported=False,
        note="算子 Profile 就绪，但通信带宽表缺失，暂不能给出完整估算。",
    ),
)


def describe_stacks() -> list[dict[str, Any]]:
    return [s.to_dict() for s in SOFTWARE_STACKS]
