"""Independent host/cgroup memory guard; standard library only, never probes GPU."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path


def linux_memory_snapshot(root=Path('/')):
    root = Path(root)
    fields = {}
    for line in (root / 'proc/meminfo').read_text(encoding='ascii').splitlines():
        name, raw = line.split(':', 1)
        fields[name] = int(raw.split()[0]) * 1024
    available = fields.get('MemAvailable')
    if available is None:
        available = fields['MemFree'] + fields.get('Buffers', 0) + fields.get('Cached', 0)
    total = fields['MemTotal']
    cgroup = None
    for limit_name, current_name in [('sys/fs/cgroup/memory.max', 'sys/fs/cgroup/memory.current'),
                                     ('sys/fs/cgroup/memory/memory.limit_in_bytes', 'sys/fs/cgroup/memory/memory.usage_in_bytes')]:
        limit_path, current_path = root / limit_name, root / current_name
        if limit_path.is_file() and current_path.is_file():
            raw = limit_path.read_text(encoding='ascii').strip()
            if raw != 'max':
                limit, current = int(raw), int(current_path.read_text(encoding='ascii').strip())
                if 0 < limit < 2**60:
                    cgroup = dict(limit_bytes=limit, used_bytes=current,
                                  available_bytes=max(0, limit - current), source=str(limit_path))
                    break
    effective = min(available, cgroup['available_bytes']) if cgroup is not None else available
    return dict(platform='linux', total_bytes=total, host_available_bytes=available,
                effective_available_bytes=effective, cgroup=cgroup, gpu_probed=False)


def memory_snapshot():
    if os.name == 'nt':
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                       ('total_physical', ctypes.c_ulonglong), ('available_physical', ctypes.c_ulonglong),
                       ('total_pagefile', ctypes.c_ulonglong), ('available_pagefile', ctypes.c_ulonglong),
                       ('total_virtual', ctypes.c_ulonglong), ('available_virtual', ctypes.c_ulonglong),
                       ('available_extended_virtual', ctypes.c_ulonglong)]
        value = MemoryStatus()
        value.length = ctypes.sizeof(value)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
            raise OSError('GlobalMemoryStatusEx failed')
        return dict(platform='windows', total_bytes=int(value.total_physical),
                    host_available_bytes=int(value.available_physical),
                    effective_available_bytes=int(value.available_physical), cgroup=None, gpu_probed=False)
    if Path('/proc/meminfo').is_file():
        return linux_memory_snapshot()
    raise RuntimeError('unsupported platform for read-only host memory guard')


def memory_reason(snapshot, minimum_free_gib=4.):
    if minimum_free_gib < 4.:
        raise ValueError('diagnostic memory floor cannot be below 4GiB')
    available = snapshot.get('effective_available_bytes')
    if not isinstance(available, int) or available < 0:
        return 'memory snapshot missing effective available bytes'
    if available < minimum_free_gib * 2**30:
        return 'effective host/cgroup memory available is below the declared diagnostic floor'
    return None
