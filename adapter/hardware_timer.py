"""Pure device execution intervals; no host spans, gaps, or cache-flush work."""
import ctypes
import json
import statistics
import struct
from pathlib import Path
from capture_support import capture_profile
from profile_parser import _resolve_kernel_names
from guard import require_reservation


def union_ns(intervals):
    ordered = sorted(set(intervals))
    if not ordered:
        return 0
    begin, end = ordered[0]
    total = 0
    for lo, hi in ordered[1:]:
        if lo <= end:
            end = max(end, hi)
        else:
            total += end - begin
            begin, end = lo, hi
    return total + end - begin


def parse_hardware(path):
    path = Path(path)
    names = _resolve_kernel_names(path)
    assert names and all(not n.startswith('<unknown') for n in names.values()), names
    assert not any('cache_flush' in n for n in names.values()), names
    cores = {}
    for f in path.rglob('ffts_profile*.slice_*'):
        if f.name.endswith('.done'):
            continue
        data = f.read_bytes()
        assert len(data) % 128 == 0
        for v in struct.iter_unpack('<16q', data):
            if not v[1]:
                continue
            seq = v[0] >> 32 & 65535
            assert v[15] >= v[14]
            cores.setdefault(seq, set()).add((v[14], v[15]))
    stars = {}
    for f in path.rglob('stars_soc.data.*.slice_*'):
        if f.name.endswith('.done'):
            continue
        info = json.loads((f.parent.parent/'info.json.0').read_text())
        assert str(info['platform_version']) == '15'
        mhz = float(info['DeviceInfo'][0]['hwts_frequency'])
        assert mhz > 0
        data = f.read_bytes()
        assert len(data) % 32 == 0
        pending = {}
        for v in struct.iter_unpack('<4HQHH3L', data):
            assert v[1] == 0x6BD3
            func = v[0] & 63
            if func not in (0, 1):
                continue
            seq, kind = v[2] + (v[3] << 16), v[0] >> 10
            key = seq, kind
            if func == 0:
                assert key not in pending
                pending[key] = v[4]
            else:
                start = pending.pop(key)
                assert v[4] >= start
                interval = (start * 1000 / mhz, v[4] * 1000 / mhz)
                stars.setdefault(seq, []).append((kind, interval))
        assert not pending
    rows, core_intervals, cpu_intervals = [], [], []
    for seq, name in names.items():
        if seq in cores:
            intervals = sorted(cores[seq])
            core_intervals.extend(intervals)
            # Verify clock alignment against enclosing hardware task timestamps.
            assert seq in stars, ('missing hardware task', seq)
            for lo, hi in intervals:
                assert any(a - 1000 <= lo <= hi <= b + 1000 for _, (a, b) in stars[seq]), (seq, intervals, stars[seq])
            kind = 'AI_CORE'
        else:
            assert seq in stars, ('uncovered named task', seq, name)
            assert all(k in (4, 5) for k, _ in stars[seq]), (seq, name, stars[seq])
            intervals = [i for _, i in stars[seq]]
            cpu_intervals.extend(intervals)
            kind = 'AICPU'
        rows.append({'sequence':seq, 'name':name, 'kind':kind,
                     'intervals_ns':intervals, 'busy_us':union_ns(intervals)/1000})
    assert set(cores) <= set(names), ('unnamed FFTS work', set(cores)-set(names))
    all_intervals = core_intervals + cpu_intervals
    assert all_intervals
    return {'device_union_us':union_ns(all_intervals)/1000,
            'ai_core_union_us':union_ns(core_intervals)/1000,
            'aicpu_union_us':union_ns(cpu_intervals)/1000,
            'device_span_us':(max(b for _, b in all_intervals)-min(a for a, _ in all_intervals))/1000,
            'tasks':rows, 'raw_path':str(path)}


class Timer:
    def __init__(self, output):
        import torch
        from tilelang.profiler.msprof import get_msprof_cache_flush_kernel
        self.torch = torch
        self.root = Path(output)
        self.root.mkdir(parents=True, exist_ok=True)
        self.flush = get_msprof_cache_flush_kernel((1 << 30)//4)
        self.storage = torch.zeros((1 << 30)//4, dtype=torch.int32, device='npu')
        (self.root/'flush.asc').write_text(self.flush.get_kernel_source())
        class Info(ctypes.Structure):
            _fields_ = [('rated',ctypes.c_uint),('current',ctypes.c_uint)]
        self.Info = Info
        self.driver = ctypes.CDLL('/usr/local/Ascend/driver/lib64/driver/libdrvdsmi_host.so')
        self.driver.dsmi_get_aicore_info.argtypes = [ctypes.c_int,ctypes.POINTER(Info)]
        self.driver.dsmi_get_aicore_info.restype = ctypes.c_int

    def frequency(self):
        values=[]
        for component in (0,1):
            info=self.Info()
            rc=self.driver.dsmi_get_aicore_info(component << 16,ctypes.byref(info))
            assert rc == 0 and 0 < info.current <= info.rated
            values.append(info.current)
        return values

    def measure(self, fn, folder, repeats=15):
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
        t=self.torch
        for _ in range(3):
            self.flush(self.storage);t.npu.synchronize()
            result=fn();t.npu.synchronize();del result
        samples=[]
        for i in range(repeats):
            require_reservation()
            assert t.npu.get_device_limit(0)=={'cube_core_num':32,'vector_core_num':64}
            self.flush(self.storage);t.npu.synchronize()
            before=self.frequency()
            with capture_profile(folder/f'profile-{i:02d}') as prof:
                result=fn();t.npu.synchronize();prof.step()
            del result
            after=self.frequency()
            raw=Path(prof.prof_if.prof_path)
            sample=parse_hardware(raw)
            sample.update(index=i,frequency_before=before,frequency_after=after)
            samples.append(sample)
            (folder/'samples.json').write_text(json.dumps(samples,indent=2)+'\n')
        result={'status':'pending_guard_receipt','protocol':'complete-hardware-union-32x64-v1',
                'warmup':3,'repeats':repeats,'flush_bytes':1<<30,'samples':samples}
        for key in ('device_union_us','ai_core_union_us','aicpu_union_us','device_span_us'):
            result['mean_'+key]=statistics.mean(s[key] for s in samples)
        (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
