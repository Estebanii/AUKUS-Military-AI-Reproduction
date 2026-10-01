#!/usr/bin/env python3
"""Table 9, Section V.5 and footnote 85: the external-control analysis (Singapore and Canada).

The two design simulations are read from the data bundle; --recompute-design recomputes them (10-15 minutes per
encoder) and --all-encoders runs the full chain for the alternative encoders too. Needs scripts 00, 03 and 05.
Writes results/external/ and results/tables/table9_external_control.csv.
"""
import _common  # noqa: F401

import argparse

from replication import data, output
from replication.external import driver


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--recompute-design', action='store_true',
                    help='recompute the design-only simulations instead of reading the archived records')
    ap.add_argument('--all-encoders', action='store_true',
                    help='run the full chain for the four alternative encoders too')
    ap.add_argument('--allow-missing-r', action='store_true',
                    help='run without R / HonestDiD (the HonestDiD blocks are then recorded as unavailable; Table 9 '
                         'itself does not use them)')
    args = ap.parse_args(argv)
    timer = output.Timer('07_external_control_table9')
    r_env = driver.r_environment()
    if r_env['ok']:
        print(f'[07] R for HonestDiD: {r_env["rscript"]} {r_env["versions"]}', flush=True)
    elif not args.allow_missing_r:
        print(f'[07] R with HonestDiD and jsonlite is needed for the event-study sensitivity ({r_env["rscript"]}: '
              f'{r_env["error"]}). Install them (see README) or rerun with --allow-missing-r.', flush=True)
        return 2
    checks = {}
    full = [data.BASELINE] + (list(data.ALTERNATIVES) if args.all_encoders else [])
    summaries, mdes, mains = {}, {}, {}
    for encoder in data.ENCODERS:
        print(f'[07] {encoder}', flush=True)
        panel, record, documents = driver.build_panel(encoder)
        if encoder in full:
            design = driver.run_design(encoder, panel, recompute=args.recompute_design)
            if args.recompute_design:
                checks[encoder] = driver.archive_matches(encoder, design)
                print(f'[07] {encoder}: recomputed design records equal the archived ones: {checks[encoder]}',
                      flush=True)
            summaries[encoder] = driver.run_estimate(encoder, panel, record, documents, design)
            mdes[encoder] = design['mde']
            if encoder != data.BASELINE:
                mains[encoder] = output.read_json(f'{driver.OUT}/{encoder}/estimate/main.json')
        else:
            mains[encoder] = driver.run_main_only(encoder, panel, record)
    table = driver.table9(summaries[data.BASELINE], mdes[data.BASELINE], mains)
    output.write_text('tables/table9_external_control.csv', table.to_csv(index=False))
    output.write_json(f'{driver.OUT}/table9.json',
                      {'rows': table.to_dict(orient='records'),
                       'design_records': 'recomputed' if args.recompute_design else 'archived (data bundle)',
                       'recomputed_equals_archived': checks or None,
                       'full_chain': full, 'r_environment': r_env})
    if checks and not all(all(v.values()) for v in checks.values()):
        print('[07] WARNING: a recomputed design record differs from the archived one (see table9.json)', flush=True)
    timer.done()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
