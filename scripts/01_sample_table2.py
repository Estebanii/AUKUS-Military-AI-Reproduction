#!/usr/bin/env python3
"""Table 2 and the sample counts of Section IV.3: occurrences, documents and year-month span by country and period
(pre-signing = year-month <= 2021-08; post = from 2021-09, month-level coding), the number of target terms, and the
pre-signing Australian occurrences. Writes results/table2_sample.json and results/tables/table2_sample.csv."""
import _common  # noqa: F401

import pandas as pd

from replication import data, output

COUNTRIES = ('US', 'UK', 'AU')


def ym(year, month) -> str:
    return f'{int(year):04d}-{int(month):02d}'


def cell(df) -> dict:
    code = df['year'] * 100 + df['month']
    first, last = int(code.min()), int(code.max())
    return {'occurrences': int(len(df)), 'documents': int(df['doc_id'].nunique()),
            'first_ym': ym(first // 100, first % 100), 'last_ym': ym(last // 100, last % 100),
            'n_years': int(df['year'].nunique())}


def main() -> int:
    timer = output.Timer('01_sample_table2')
    df = data.occurrence_meta(['country', 'year', 'month', 'post_aukus', 'doc_id', 'term'])
    code = df['year'] * 100 + df['month']
    if not bool(((code >= 202109) == df['post_aukus'].astype(bool)).all()):
        raise SystemExit('post_aukus disagrees with year-month >= 2021-09')
    per_doc_country = df.groupby('doc_id')['country'].nunique()
    per_doc_period = df.groupby('doc_id')['post_aukus'].nunique()
    out = {'definitions': {'pre': 'post_aukus == False (year-month <= 2021-08)',
                           'post': 'post_aukus == True (year-month >= 2021-09, month-level coding)',
                           'documents': 'distinct doc_id (the cluster unit of the bootstrap)'},
           'total': {**cell(df), 'terms': int(df['term'].nunique())},
           'docs_in_more_than_one_country': int((per_doc_country > 1).sum()),
           'docs_in_both_periods': int((per_doc_period > 1).sum()), 'by_country': {}, 'by_period': {}}
    for period, mask in (('pre', ~df['post_aukus'].astype(bool)), ('post', df['post_aukus'].astype(bool))):
        out['by_period'][period] = cell(df[mask])
    rows = []
    for c in COUNTRIES:
        dc = df[df['country'] == c]
        block = {'all': {**cell(dc), 'share_of_occurrences': len(dc) / len(df)}}
        for period, mask in (('pre', ~dc['post_aukus'].astype(bool)), ('post', dc['post_aukus'].astype(bool))):
            block[period] = cell(dc[mask])
        out['by_country'][c] = block
        for period in ('pre', 'post', 'all'):
            rows.append({'country': c, 'period': period, **{k: v for k, v in block[period].items()}})
    au_pre = df[(df['country'] == 'AU') & ~df['post_aukus'].astype(bool)]
    out['au_pre'] = {'occurrences': int(len(au_pre)), 'in_2021': int((au_pre['year'] == 2021).sum()),
                     'share_in_2021': float((au_pre['year'] == 2021).mean())}
    out['corpus'] = {'documents_all': int(df['doc_id'].nunique()), 'terms_listed': int(len(data.terms()))}
    output.write_json('table2_sample.json', out)
    output.write_text('tables/table2_sample.csv', pd.DataFrame(rows).to_csv(index=False))
    timer.done()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
