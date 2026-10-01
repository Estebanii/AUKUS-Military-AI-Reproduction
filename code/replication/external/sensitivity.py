"""The sensitivity items (descriptive; raw month-block p values reported, never members of a test family).

Each item is a :class:`replication.external.estimate.Spec` variant of the main M1 specification of its sample, estimated on
every response column with the circular month-block bootstrap (same seed and draws per calendar):

SG to 2023-12; donut (2021-09..2021-12 removed from data and calendar); day-level cut 2021-09-16 (Post, strata and
the 2021-09 split all at the day); news + release genres only; fixed genre composition (pre-period shares per
country; may change the estimand); document equal weight; concurrent events (SG documents 2021-08-15..2021-09-30
related to cyber cooperation removed: title or content matching ``cyber``, case-insensitive; the removed documents
are listed); M3 term FE; the 2017+ shrink window; the 2017-09+ common window (all four countries observed); without
AU (AU also enters the common trend and year effects); SG + CA pooled (one control group, 2017-01..2025-11); without
the cyber terms; block length L = 3 and 12. The cross-control comparison (SG vs CA) is read on the same window
(S_SG 2017+ vs S_CA 2017+).
"""
from __future__ import annotations

import re
from dataclasses import replace

import pandas as pd

from . import estimate as est
from .params import external_params


def concurrent_documents(docs: pd.DataFrame) -> pd.DataFrame:
    """SG documents dated in the window whose title or content matches the cyber pattern."""
    s = external_params('sensitivity')['concurrent_sg']
    d = docs.loc[docs['country'].eq('SG')]
    date = d['publish_date'].astype(str)
    inside = (date >= s['start']) & (date <= s['end'])
    pattern = re.compile(s['pattern'], re.I)
    hit = d[s['fields']].fillna('').astype(str).apply(lambda row: any(pattern.search(v) for v in row), axis=1)
    return d.loc[inside & hit, ['article_id', 'publish_date', 'title']].reset_index(drop=True)


def items(sample: str, concurrent_ids: tuple) -> dict:
    """The specifications of one sample (SG or CA), keyed by item name."""
    s = external_params('sensitivity')
    w = external_params('windows')
    base = est.main_spec(sample)
    out = {}
    if sample == 'SG':
        out['SG_to_2023_12'] = replace(base, name='SG_to_2023_12', end=w['SG_to_2023_12']['end'],
                                       description='SG 截断至 2023-12')
    out['donut'] = replace(base, name=f'donut_{sample}', drop_months=tuple(s['donut']),
                           description='donut：剔除 2021-09…2021-12')
    out['day_cut'] = replace(base, name=f'day_cut_{sample}', cut='day', description='日级截止 2021-09-16')
    out['news_release_only'] = replace(base, name=f'news_release_{sample}', genres=tuple(s['genres']),
                                       description='仅 news + news release 体裁')
    out['genre_fixed_weights'] = replace(base, name=f'genre_fixed_{sample}', weights='genre_fixed',
                                         description='固定体裁构成重加权（按签署前各国体裁份额；可能改变估计对象）')
    out['document_weights'] = replace(base, name=f'document_weights_{sample}', weights='document',
                                      description='文档等权')
    if sample == 'SG':
        out['concurrent_events'] = replace(base, name='concurrent_SG', drop_docs=tuple(concurrent_ids),
                                           description='同期事件：剔除 SG 2021-08-15…2021-09-30 网络合作相关文档')
    out['M3_term_FE'] = replace(base, name=f'M3_{sample}', model='M3', description='M3 术语 FE')
    if sample == 'SG':
        out['shrink_2017'] = replace(base, name='shrink_2017_SG', start=w['shrink_2017']['start'],
                                     description='2017+ 收缩窗口（与 S_CA 同窗口的跨对照比较）')
    out['common_2017_09'] = replace(base, name=f'common_2017_09_{sample}', start=w['common_2017_09']['start'],
                                    description='2017-09+ 共同历时窗口')
    out['drop_AU'] = replace(base, name=f'drop_AU_{sample}', drop_countries=('AU',), description='去 AU')
    out['drop_cyber'] = replace(base, name=f'drop_cyber_{sample}', drop_terms=tuple(s['cyber_terms']),
                                description='剔除 cyber 术语')
    for L in external_params('bootstrap')['block_length_sensitivity']:
        out[f'block_L{L}'] = replace(base, name=f'block_L{L}_{sample}', block_length=L, description=f'块长 L = {L}')
    return out


def pooled() -> est.Spec:
    w = external_params('windows')['pooled']
    return est.Spec(name='pooled_SG_CA', sample='SG+CA', start=w['start'], end=w['end'],
                    description='SG + CA 合并为一个对照组（2017-01…2025-11）')


def run(panel: pd.DataFrame, responses: list, cache: est.DrawCache, concurrent: pd.DataFrame, progress=None) -> dict:
    out = {'SG': {}, 'CA': {}, 'pooled': {}}
    ids = tuple(concurrent['article_id'])
    for sample in ('SG', 'CA'):
        for name, spec in items(sample, ids).items():
            if progress:
                progress(f'sensitivity {sample} {name}')
            out[sample][name] = {'description': spec.description, **est.fit_spec(panel, spec, responses, cache)}
    spec = pooled()
    out['pooled']['SG_CA'] = {'description': spec.description, **est.fit_spec(panel, spec, responses, cache)}
    out['concurrent_documents_removed'] = concurrent.to_dict('records')
    return out
