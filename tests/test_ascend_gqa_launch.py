"""Host launch planning must follow device capacity without importing an NPU backend."""

import dataclasses
import math
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import tileops

pytestmark = pytest.mark.smoke

_CONFIG = runpy.run_path(
    str(Path(tileops.__file__).parent / "kernels" / "ascend" / "attention" / "config.py")
)


@dataclasses.dataclass(frozen=True)
class LaunchCall:
    batch: int = 1
    heads: int = 32
    heads_kv: int = 4
    dim: int = 128
    max_seqlen_q: int = 1
    seqlen_kv: int = 8192
    dtype: torch.dtype = torch.float16
    is_causal: bool = True
    sm_scale: float | None = None
    softcap: float = 0.0
    window_size_left: int = -1
    window_size_right: int = -1
    is_fp8: bool = False
    fuse_rope: bool = False
    rotary_dim: int = 64
    rope_layout: str = "neox"
    device: object = None


@pytest.mark.parametrize("cores", [1, 2, 8, 23, 24, 32, 36, 48, 64])
@pytest.mark.parametrize(
    "call",
    [
        LaunchCall(),
        LaunchCall(seqlen_kv=4096),
        LaunchCall(seqlen_kv=262144),
        LaunchCall(batch=32, heads_kv=8, seqlen_kv=4096),
        LaunchCall(batch=16, heads=48, heads_kv=1),
        LaunchCall(heads=8, heads_kv=1, dim=256),
        LaunchCall(batch=16, heads=8, heads_kv=1, dim=256),
    ],
)
def test_decode_capacity_and_partition_coverage(cores, call):
    variant, plan = _CONFIG["select_kernel"](call, cores)
    assert variant == "decode-multibuffer"
    grid, splits = plan["cores"], plan["splits"]
    assert 0 < grid <= cores
    tasks = (
        call.batch
        * call.heads_kv
        * math.ceil((call.heads // call.heads_kv) / plan["query_block"])
        * splits
    )
    assigned = [task for block in range(grid) for task in range(block, tasks, grid)]
    assert sorted(assigned) == list(range(tasks))
    tiles = call.seqlen_kv // plan["kv_block"]
    partitions = [range(s * tiles // splits, (s + 1) * tiles // splits) for s in range(splits)]
    assert all(partitions)
    assert [tile for part in partitions for tile in part] == list(range(tiles))
    assert max(map(len, partitions)) - min(map(len, partitions)) <= 1
    assert plan.get("grid_fuse", False) == (splits > 1)
    if splits > 1:
        assert grid == cores
    else:
        assert plan["persistent"]


def test_split_selection_tracks_capacity_and_cache_key():
    call = LaunchCall()
    assert [_CONFIG["select_kernel"](call, c)[1]["splits"] for c in (24, 32, 36, 48)] == [
        6,
        8,
        9,
        12,
    ]
    assert _CONFIG["configuration"](call, 8)[2] == 4
    assert _CONFIG["configuration"](call, 48)[2] == 24
    assert _CONFIG["configuration"](call, 8)[2] == 4


@pytest.mark.parametrize("cores", [8, 23, 36, 48])
@pytest.mark.parametrize(
    "call",
    [
        LaunchCall(heads=8, heads_kv=2, max_seqlen_q=256, seqlen_kv=1792, is_fp8=True),
        LaunchCall(heads=8, heads_kv=2, max_seqlen_q=256, seqlen_kv=1792, fuse_rope=True),
        LaunchCall(heads_kv=8, max_seqlen_q=128, seqlen_kv=2048, dtype=torch.bfloat16),
    ],
)
def test_prefill_receives_device_capacity(cores, call):
    _, plan = _CONFIG["select_kernel"](call, cores)
    assert plan["num_cores"] == cores


def test_device_query_uses_operand_device(monkeypatch):
    seen = []

    def properties(device):
        seen.append(device)
        return SimpleNamespace(cube_core_num=24 if device == "npu:1" else 48)

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(current_device=lambda: 0, get_device_properties=properties),
        raising=False,
    )
    query = _CONFIG["device_cube_cores"]
    assert query(LaunchCall(device="npu:1")) == 24
    assert query(LaunchCall()) == 48
    assert seen == ["npu:1", 0]


@pytest.mark.parametrize("cores", [0, -1])
def test_invalid_capacity_fails_before_launch(cores, monkeypatch):
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            current_device=lambda: 0,
            get_device_properties=lambda _: SimpleNamespace(cube_core_num=cores),
        ),
        raising=False,
    )
    with pytest.raises(RuntimeError, match="positive"):
        _CONFIG["device_cube_cores"](LaunchCall())
    with pytest.raises(ValueError, match="positive"):
        _CONFIG["select_kernel"](LaunchCall(), cores)
