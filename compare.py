#!/usr/bin/env python3
"""Check the recomputed results against every number printed in the quantitative part of the manuscript.

For each of the items in ``expected/paper_numbers.csv`` (one row per printed number, with its location in the
manuscript) the value is read from the results (or from the data bundle, a fixed parameter, or an expression over
these), rounded as the manuscript rounds it (ROUND_HALF_UP on the shortest decimal representation, the printed number
of decimals; percentages x 100) and compared with the printed number. Significance stars are checked against the
bootstrap p value. Rules: ``round`` (the rounded value equals the printed one), ``p_floor`` (printed "<0.001": the
value is below 0.001), ``threshold`` (a printed threshold equals the parameter), ``text`` (year-month and similar text
equal), ``label`` (a printed label, not a computed number; listed, not tested).

Every item is also compared with the full-precision reference value of the manuscript's own results ("exact" when
identical to the last bit). In the pinned environment every recomputed item is exact; on another platform or BLAS the
last digits may differ, which the display-precision test (PASS / FAIL) tolerates by construction.

Usage: python compare.py [--results DIR] [--data DIR]. Writes results/compare/items.csv and
results/compare/summary.csv; prints the run record of results/run_info.json (written by run_all.sh) first and one
line per manuscript table, figure, footnote and text section; exit code 1 if any item FAILS, 2 if the run record is
missing or its package commit (git HEAD, or the commit id in VERSION when there is no .git), code fingerprint or data
manifest differs from the current copy of the package.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE / 'code'))
ALTERNATIVES = ('ModernBERT-large', 'roberta-large', 'ettin-encoder-1b', 'deberta-v2-xlarge')
_CACHE: dict = {}


# --------------------------------------------------------------------------- reading values
def _json(path: Path):
    if path not in _CACHE:
        _CACHE[path] = json.loads(path.read_text(encoding='utf-8'))
    return _CACHE[path]


def resolve(obj, pointer: str):
    for part in [p for p in pointer.split('/') if p]:
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


def param(dotted: str):
    """A fixed parameter: 'section.key.index' of replication.params, or 'external:section.key' of the external-control
    parameters."""
    if dotted.startswith('external:'):
        from replication.external.params import EXTERNAL_PARAMS as table
        dotted = dotted[len('external:'):]
    else:
        from replication.params import PARAMS as table
    return resolve(table, dotted.replace('.', '/'))


def max_year_key(d: dict) -> int:
    return max(int(k.split('_')[-1]) for k in d if k.startswith('UK_x_'))


def min_year_key(d: dict) -> int:
    return min(int(k.split('_')[-1]) for k in d if k.startswith('UK_x_'))


def year_of_rank(coefficients: dict, years, rank: int) -> int:
    """The year whose UK x year coefficient has the given rank (1 = largest) among ``years``."""
    ordered = sorted(years, key=lambda y: coefficients[f'UK_x_{y}']['coef'], reverse=True)
    return int(ordered[rank - 1])


TRANSFORMS = {
    '': lambda v: v,
    'len': len,
    'min_year_key': min_year_key,
    'max_year_key': max_year_key,
    'month_code_ym': lambda v: f'{int(v) // 100:04d}-{int(v) % 100:02d}',
    'month_code_year': lambda v: int(v) // 100,
    'ym_year': lambda v: int(str(v)[:4]),
    'mde_threshold': lambda v: float(re.match(r'MDE > ([0-9.]+) SD', v).group(1)),
}


def recompute(row: dict, results: Path, bundle: Path | None):
    """The recomputed value of one item (before scaling)."""
    results = Path(results)
    bundle = Path(bundle) if bundle else None

    def v(file, pointer=''):
        return resolve(_json(results / file), pointer)

    def b(file, pointer=''):
        return resolve(_json(bundle / file), pointer)
    source = row['source']
    if source == 'results':
        value = v(row['file'], row['pointer'])
    elif source == 'bundle':
        if row['transform'] == 'npy_width':
            import numpy as np
            return int(np.load(bundle / row['file'], mmap_mode='r').shape[1])
        value = b(row['file'], row['pointer'])
    elif source == 'param':
        return param(row['expression'])
    elif source == 'expression':
        names = {'v': v, 'b': b, 'param': param, 'abs': abs, 'min': min, 'max': max, 'sum': sum, 'int': int,
                 'ALTERNATIVES': ALTERNATIVES, 'max_year_key': max_year_key, 'min_year_key': min_year_key,
                 'year_of_rank': year_of_rank}
        return eval(row['expression'], {'__builtins__': {}, **names})  # expressions of expected/paper_numbers.csv only
    else:
        raise ValueError(f'{row["item_id"]}: source {source!r} is not computed')
    return TRANSFORMS[row['transform']](value)


# --------------------------------------------------------------------------- display comparison
def round_half_up(x, decimals: int) -> Decimal:
    d = Decimal(x) if isinstance(x, int) else Decimal(repr(float(x)))
    return d.quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)


def printed_number(text: str) -> Decimal:
    clean = (text.replace('−', '-').replace(',', '').replace('%', '').replace('(', '').replace(')', '')
             .replace('<', '').replace('*', '').replace('†', '').strip())
    return Decimal(clean)


def stars(p: float) -> str:
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '†' if p < 0.1 else ''


def check(row: dict, value) -> tuple:
    """(status, recomputed display, detail)."""
    rule, printed = row['rule'], row['printed_display']
    if rule == 'label':
        return 'LABEL', '', 'printed label, not a computed number'
    if rule == 'text':
        shown = str(value)
        return ('PASS' if shown == row['printed_value'] else 'FAIL'), shown, ''
    scaled = value * int(row['scale']) if row['scale'] not in ('', '1') else value
    if rule == 'p_floor':
        return ('PASS' if float(scaled) < 0.001 else 'FAIL'), f'{float(scaled):.3g}', 'printed <0.001'
    if rule == 'threshold':
        ok = Decimal(repr(float(scaled))) == printed_number(printed)
        return ('PASS' if ok else 'FAIL'), repr(float(scaled)), 'printed threshold'
    decimals = int(row['decimals'] or 0)
    shown = round_half_up(scaled, decimals)
    return ('PASS' if shown == printed_number(printed) else 'FAIL'), str(shown), ''


def exactness(row: dict, value) -> str:
    ref = row['reference_value']
    if row['rule'] == 'label' or ref == '':
        return 'n/a'
    if isinstance(value, str) or row['rule'] == 'text':
        return 'exact' if str(value) == ref else 'differs'
    scaled = float(value) * float(row['scale'] or 1)
    if scaled == float(ref):
        return 'exact'
    return f'differs by {abs(scaled - float(ref)):.2g}'


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results', type=Path, default=Path(os.environ.get('REPLICATION_RESULTS') or PACKAGE / 'results'))
    ap.add_argument('--data', type=Path,
                    default=Path(os.environ.get('REPLICATION_DATA') or PACKAGE / 'data' / 'bundle'))
    ap.add_argument('--expected', type=Path, default=PACKAGE / 'expected' / 'paper_numbers.csv')
    args = ap.parse_args(argv)
    from replication import run_info
    record, run_problems = run_info.check(args.results, args.data)
    if record:
        print(f'Results of run {record["run_id"]} (package commit {record.get("package_commit")} from '
              f'{record.get("package_commit_source")}, code '
              f'{record["code_sha256"][:12]}, data manifest {record["data_manifest_sha256"]})')
    for problem in run_problems:
        print(f'WARNING: {problem}')
    rows = list(csv.DictReader(open(args.expected, encoding='utf-8')))
    out, summary = [], {}
    for row in rows:
        try:
            value = None if row['rule'] == 'label' else recompute(row, args.results, args.data)
            status, shown, detail = check(row, value)
            if status == 'PASS' and row['stars_pointer']:
                p = resolve(_json(args.results / row['file']), row['stars_pointer'])
                want = re.sub(r'[^*†]', '', row['printed_display'])
                if stars(float(p)) != want:
                    status, detail = 'FAIL', f'stars {stars(float(p))!r} vs printed {want!r}'
                else:
                    detail = f'stars {want or "none"} match p = {float(p):.3f}'
            exact = 'n/a' if value is None else exactness(row, value)
        except FileNotFoundError as err:
            status, shown, detail, exact = 'FAIL', '', f'missing file: {err.filename} (run the scripts first)', 'n/a'
        except (KeyError, IndexError, TypeError, ValueError) as err:
            status, shown, detail, exact = 'FAIL', '', f'{type(err).__name__}: {err}', 'n/a'
        out.append({'item_id': row['item_id'], 'paper_item': row['paper_item'], 'status': status,
                    'printed': row['printed_display'], 'recomputed_display': shown, 'full_precision': exact,
                    'detail': detail, 'source': row['source'], 'file': row['file'] or row['expression'],
                    'pointer': row['pointer'], 'location': row['location']})
        s = summary.setdefault(row['paper_item'], {'PASS': 0, 'FAIL': 0, 'LABEL': 0, 'exact': 0, 'noref': 0})
        s[status] += 1
        s['exact'] += exact == 'exact'
        s['noref'] += status != 'LABEL' and row['reference_value'] == ''
    target = args.results / 'compare'
    target.mkdir(parents=True, exist_ok=True)
    with open(target / 'items.csv', 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    with open(target / 'summary.csv', 'w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['paper_item', 'status', 'items', 'pass', 'fail', 'labels', 'exact_full_precision',
                         'no_full_precision_reference'])
        for item, s in summary.items():
            n = s['PASS'] + s['FAIL'] + s['LABEL']
            writer.writerow([item, 'FAIL' if s['FAIL'] else 'PASS', n, s['PASS'], s['FAIL'], s['LABEL'], s['exact'],
                             s['noref']])
    width = max(len(k) for k in summary)
    for item, s in summary.items():
        n = s['PASS'] + s['FAIL'] + s['LABEL']
        label = f', {s["LABEL"]} label(s)' if s['LABEL'] else ''
        noref = f' ({s["noref"]} without a full-precision reference)' if s['noref'] else ''
        print(f'{"FAIL" if s["FAIL"] else "PASS"}  {item:<{width}}  {s["PASS"]}/{n - s["LABEL"]} items{label}; '
              f'{s["exact"]} identical to the reference at full precision{noref}')
    fails = [r for r in out if r['status'] == 'FAIL']
    for r in fails:
        print(f'  FAIL {r["item_id"]} ({r["location"][:70]}): printed {r["printed"]}, recomputed '
              f'{r["recomputed_display"] or "-"}; {r["detail"]}')
    total = len(out)
    n_pass = sum(r['status'] == 'PASS' for r in out)
    n_label = sum(r['status'] == 'LABEL' for r in out)
    n_exact = sum(r['full_precision'] == 'exact' for r in out)
    n_noref = sum(s['noref'] for s in summary.values())
    print(f'\n{n_pass} of {total - n_label} printed numbers reproduced at the printed precision ({len(fails)} FAIL, '
          f'{n_label} labels not tested); {n_exact} identical to the reference at full precision, {n_noref} without a '
          f'full-precision reference (see expected/paper_numbers.csv, column note). Details: {target / "items.csv"}')
    status = 1 if fails else (2 if run_problems else 0)
    if run_problems:
        print(f'\nWARNING: the results may not come from a fresh run of this copy of the package (exit status {status}'
              + (', set by the FAIL items' if fails else '') + '):')
        for problem in run_problems:
            print(f'WARNING: {problem}')
    return status


if __name__ == '__main__':
    raise SystemExit(main())
