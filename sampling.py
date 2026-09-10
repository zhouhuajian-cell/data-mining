# -*- coding: utf-8 -*-
# sampling.py
import numpy as np

DEFAULT_SAMPLE_COUNT = 5


def sample_representative(frame_paths, count=DEFAULT_SAMPLE_COUNT):
    """方案第4节：均匀取样 F1/F13/F25/F37/F50，观察整个 Clip 的变化"""
    n = len(frame_paths)
    if n == 0:
        return []
    idx = np.linspace(0, n - 1, min(count, n)).round().astype(int)
    # 去重保序（n<count 时 linspace 会重复）
    seen, out = set(), []
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(frame_paths[i])
    return out
