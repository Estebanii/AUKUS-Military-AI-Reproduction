"""The low-dimensional pre-trend diagnostic (the pre-trend inference of the external-control analysis).

Pre-signing rows only (members + the control, the sample's main window up to 2021-08); model 'slope' of
:mod:`replication.external.estimate`: Y = a + sum_g gamma_g D_g + beta t + sum_g phi_g D_g t, t = the linear calendar-month
trend in years since 2014-01. Estimands: slope_US, slope_UK, slope_AU (phi_g: member g's linear pre-signing slope
minus the control's) and theta_slope = slope_UK - slope_US. Inference: the M1 synchronised month-block bootstrap
(:func:`replication.external.estimate.fit_spec`): one layer of pre-signing calendar months, circular blocks, every country
resampled together; L = 6 (L = 3 and 12 as sensitivity); B = 9,999, seed 20260928; the centred two-sided +1 p value;
the symmetric absolute-deviation interval; the fixed-priority structural columns and target-estimability
failure rule (<= 1% of B replaced, else "不可用").

The test diagnoses linear deviations only: it cannot replace period-by-period tests and cannot prove parallel trends
(Roth 2022). Its size and coverage are simulated before any estimate (:mod:`replication.external.simulation`, fixed
together with the MDE). Inference selection: the inference of theta_slope and slope_UK is the result-blind selection of
:mod:`replication.external.selection` (applied in the estimation), or descriptive on its fallback; the L = 6
month-block results are kept for comparison.
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd

from . import estimate as est
from .params import CONTROLS, external_params


def slope_spec(sample: str, block_length: int | None = None) -> est.Spec:
    """The pre-signing slope specification of ``sample`` (SG or CA main window)."""
    pt = external_params('pretrend')
    w = external_params('windows')[sample]
    return est.Spec(name=f'pretrend_slope_{sample}', sample=sample, start=w['start'], end=w['end'], model='slope',
                    pre_only=True, block_length=int(pt['block_length'] if block_length is None else block_length),
                    description='低维预趋势斜率诊断（签署前行；国家 × 线性月趋势，相对对照国）')


def pretrend(panel: pd.DataFrame, responses: list, cache: est.DrawCache, simulation: dict | None = None) -> dict:
    """The slope diagnostic of both samples: L = 6 (main) and L = 3 / 12 (sensitivity), every response column, with
    the pre-specified simulation judgements of the slope procedures (``simulation``: the coverage / size simulation record)."""
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
