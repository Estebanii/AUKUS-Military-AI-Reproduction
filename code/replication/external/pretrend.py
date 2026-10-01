"""The pre-trend slope diagnostic of the external-control analysis: linear pre-signing trends of each member
relative to the control, with month-block inference.
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd

from . import estimate as est
from .params import CONTROLS, external_params


def slope_spec(sample: str, block_length: int | None = None) -> est.Spec:
    pt = external_params('pretrend')
    w = external_params('windows')[sample]
    return est.Spec(name=f'pretrend_slope_{sample}', sample=sample, start=w['start'], end=w['end'], model='slope',
                    pre_only=True, block_length=int(pt['block_length'] if block_length is None else block_length),
                    description='低维预趋势斜率诊断（签署前行；国家 × 线性月趋势，相对对照国）')


def pretrend(panel: pd.DataFrame, responses: list, cache: est.DrawCache, simulation: dict | None = None) -> dict:
    pt = external_params('pretrend')
    out = {'decision': pt['decision'], 'statement': pt['statement'], 'rule': pt, 'samples': {}}
    for sample in CONTROLS:
        main = est.fit_spec(panel, slope_spec(sample), responses, cache)
        sens = {f'L{L}': est.fit_spec(panel, replace(slope_spec(sample), block_length=int(L),
                                                     name=f'pretrend_slope_{sample}_L{L}'), responses, cache)
                for L in pt['block_length_sensitivity']}
        sim = ((simulation or {}).get('samples') or {}).get(sample) or {}
        procs = sim.get('procedures') or {}
        out['samples'][sample] = {
            'main': main, 'sensitivity': sens,
            'simulation': {name: (procs.get(name) or {}).get('by_item') for name in ('slope', 'slope_L3', 'slope_L12')}
                          if procs else None}
    return out
