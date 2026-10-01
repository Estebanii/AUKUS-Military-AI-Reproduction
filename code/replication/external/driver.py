"""The chain of the external-control analysis for one encoder (Table 9 and footnote 85).

1. Panel: the encoder's three-country rows and its Singapore / Canada rows, measured with the encoder's A of the
   main analysis, on the reference axes and on the encoder's own PCA-3 (:mod:`.panel`).
2. Design-only records, fixed before any estimate: the choice of the M1 and pre-trend slope inference among eight
   candidates (:mod:`.selection`), the coverage / size simulation of every procedure (:mod:`.simulation`), and the
   observed-precision MDE of the selected M1 inference (:func:`mde_blocks`; the outcome enters only through the
   residuals of the SE). The two simulations use only the design (rows, countries, months, documents, weights) and
   take 10-15 minutes per encoder. By default the records archived in the data bundle are used (they are what
   :func:`selection.run` and :func:`simulation.run` return); ``recompute=True`` recomputes them.
3. Estimates: fake cut-off diagnostic, the main models M1 / M0 / M2 with the selected M1 inference,
   theta_M1 - theta_M0, the document-cluster wild cluster bootstrap beside them, the pre-trend slopes, the descriptive
   event study with its labelled sensitivity (HonestDiD through R), the placebos, the sensitivity samples and a summary.

Every record is passed on as it reads back from its JSON file (numbers exact, keys as strings).
"""
from __future__ import annotations

import json

import pandas as pd

from .. import data, output, pipeline
from . import estimate as est
from . import eventstudy, inference as inf, panel as panel_mod, pretrend, selection, sensitivity, simulation
from .params import CONTROLS, external_params

OUT = 'external'
SELECTED = 'M1_selected'
DESCRIPTIVE = 'descriptive (inference selection)'
ARCHIVE = 'external_controls/design_records'
DESIGN_FILES = ('inference_selection.json', 'simulation.json', 'mde.json')
ESTIMATE_FILES = ('fake_cutoffs.json', 'main.json', 'pretrend_slopes.json', 'event_study.json', 'placebos.json',
                  'sensitivity.json', 'summary.json')


def log(message: str) -> None:
    print(f'[external] {message}', flush=True)


def roundtrip(obj):
    """The object as it reads back from its JSON file."""
    return json.loads(json.dumps(output.plain(obj)))


# --------------------------------------------------------------------------- R (HonestDiD)
def r_environment() -> dict:
    """The R used by the HonestDiD sensitivity of the event study: {'ok', 'rscript', 'versions' or 'error'}."""
    import subprocess
    exe = eventstudy.rscript()
    code = ('suppressPackageStartupMessages({library(HonestDiD); library(jsonlite)}); '
            'cat(R.version$major, R.version$minor, as.character(packageVersion("HonestDiD")), '
            'as.character(packageVersion("jsonlite")))')
    try:
        proc = subprocess.run([exe, '-e', code], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as error:
        return {'ok': False, 'rscript': exe, 'error': str(error)}
    if proc.returncode != 0:
        return {'ok': False, 'rscript': exe, 'error': proc.stderr.strip()[-500:]}
    major, minor, honest, jsonlite = proc.stdout.split()
    return {'ok': True, 'rscript': exe, 'versions': {'R': f'{major}.{minor}', 'HonestDiD': honest,
                                                     'jsonlite': jsonlite}}


# --------------------------------------------------------------------------- inputs
def analysis_record(encoder: str) -> tuple:
    """(orientation, meta) of the encoder's main analysis (written by scripts 03 and 05)."""
    base = 'h2' if encoder == data.BASELINE else f'alternatives/{encoder}'
    return output.read_json(f'{base}/orientation.json'), output.read_json(f'{base}/meta.json')


def build_panel(encoder: str) -> tuple:
    """(panel, panel record, documents) of one encoder."""
    fit = pipeline.load_fit(encoder)
    orientation, meta = analysis_record(encoder)
    bound = {'A': fit['A'], 'U_mean': fit['U_mean'], 'y_table': pipeline.analysis_table(encoder, fit['Y']),
             'orientation': orientation, 'meta': meta, 'axes': pipeline.load_axes()}
    U, rows = data.external_encoding(encoder)
    documents = data.external_documents()
    panel, record, _, _ = panel_mod.build(bound, U, rows, data.external_targets_meta(), documents)
    return panel, record, documents


def archived(encoder: str, name: str) -> dict:
    return data.archived_json(f'{ARCHIVE}/{encoder}/{name}.json')


# --------------------------------------------------------------------------- the MDE
def family_of(encoder: str, estimand: str):
    for name, fam in external_params('families').items():
        if isinstance(fam, dict) and 'encoders' in fam and encoder in fam['encoders'] and estimand in fam['estimands']:
            return name, fam['m']
    return None, None


def _at_level(by: dict | None, a: float) -> dict:
    """The entry of a level-keyed dict (float or JSON-string keys); {} when absent."""
    for k, v in (by or {}).items():
        if abs(float(k) - a) < 1e-12:
            return v or {}
    return {}


def mde_blocks(encoder: str, panel: pd.DataFrame, sel: dict, cache: est.DrawCache | None = None) -> tuple:
    """(per-sample MDE blocks, member tiers): MDE = (c + z_.80) x SE with c the critical value of the selected M1
    inference (:func:`selection.observed_mde`) at the member family's alpha / m (the Holm first step; other rows:
    alpha); the unadjusted alpha is reported alongside and the five scenario MDEs beside it, descriptive only. No
    estimate or p enters the record."""
    alpha = external_params('mde')['alpha']
    member = external_params('families')['member_spec']
    blocks, members = {}, {}
    for sample in CONTROLS:
        spec = est.main_spec(sample)
        sd = est.locked_sd(panel, spec, panel_mod.RESPONSES)
        obs = selection.observed_mde(panel, sel, sample, panel_mod.RESPONSES, sd, cache)
        scen = selection.mde_report(panel, sel, sample, panel_mod.RESPONSES, sd)
        tiers = {}
        for name in external_params('inference_selection')['procedures']['M1']['estimands']:
            fam, m = family_of(encoder, name)
            tiers[name] = {}
            for r in panel_mod.RESPONSES:
                is_member = fam is not None and sample == member['sample'] and r == 'ref_' + member['pc']
                level = alpha / m if is_member else alpha
                o = ((obs.get('estimands') or {}).get(name) or {}).get(r) or {}
                at, una = _at_level(o.get('by_level'), level), _at_level(o.get('by_level'), alpha)
                value = at.get('mde_sd')
                entry = {'rule': 'observed_precision', 'level': level, 'member': is_member, 'family': fam,
                         'status': o.get('status') or obs.get('status'), 'se': o.get('se'), 'df': o.get('df'),
                         'c': at.get('c'), 'c_calibrated': at.get('c_calibrated'), 'mde': at.get('mde'),
                         'mde_sd': value, 'tier': inf.tier(value),
                         'unadjusted_alpha': {'level': alpha, **{k: una.get(k) for k in ('c', 'c_calibrated', 'mde',
                                                                                         'mde_sd')},
                                              'tier_reported_only': inf.tier(una.get('mde_sd'))['tier']},
                         'scenario_mde_descriptive': {
                             'level': level,
                             'mde_unit': _at_level(scen.get('mde_unit_by_level'), level).get(name),
                             'mde_sd': (_at_level(scen.get('by_level'), level).get(name) or {}).get(r)}}
                tiers[name][r] = entry
                if is_member:
                    members[f'{sample}:{name}:{r}'] = {'tier': entry['tier']['tier'], 'family': fam,
                                                       'mde_sd_family_level': value,
                                                       'mde_sd_unadjusted_alpha': una.get('mde_sd'),
                                                       'tier_unadjusted_alpha_reported_only':
                                                           entry['unadjusted_alpha']['tier_reported_only']}
        blocks[sample] = {'spec': spec.record(), 'locked_sd': sd, 'observed': obs, 'scenarios_descriptive': scen,
                          'estimands': tiers}
    return blocks, members


def simulation_verdicts(sim: dict) -> dict:
    """Per sample and procedure: the counts of 通过 / 未定 / 失败 over every scenario, estimand and metric."""
    return {sample: {name: proc.get('verdicts') for name, proc in (block.get('procedures') or {}).items()}
            for sample, block in (sim.get('samples') or {}).items()}


# --------------------------------------------------------------------------- step 2: design-only records
def run_design(encoder: str, panel: pd.DataFrame, recompute: bool = False) -> dict:
    """The inference selection, the MDE and the coverage / size simulation of one encoder (written under
    results/external/<encoder>/design/); returns the three records as read back from their files."""
    base = f'{OUT}/{encoder}/design'
    origin = 'recomputed' if recompute else f'archived in the data bundle ({ARCHIVE}/{encoder}/)'
    cache = est.DrawCache()
    if recompute:
        log(f'{encoder}: inference selection (design only; development and verification simulations)')
        sel = roundtrip(selection.run(panel, progress=log, cache=cache))
    else:
        sel = archived(encoder, 'inference_selection')
    sel_record = {'encoder': encoder, 'origin': origin, **sel, 'written_before_any_estimate': True,
                  'statement': '结果盲态选择：候选、开发与独立验证模拟、Monte Carlo 误差与选择轨迹均先于任何估计确定'
                               '（只用设计，不读取结果变量）'}
    output.write_json(f'{base}/inference_selection.json', sel_record)
    log(f'{encoder}: observed-precision MDE of the selected M1 inference (SE only, no estimate)')
    blocks, members = mde_blocks(encoder, panel, sel, cache)
    mde_record = {'encoder': encoder, 'decision': 'minimum detectable effect', 'mde': blocks, 'members': members,
                  'rule': {'mde': external_params('mde'), 'inference_selection_mde': external_params('inference_selection')['mde']},
                  'written_before_any_estimate': True, 'contains_estimates': False,
                  'statement': 'MDE = (c + z_.80) × SE：c 为选定 M1 推断在各窗口的实际临界值（C8：校准临界值 c*_α × '
                               'Satterthwaite t_.975），成员按 Holm 首步 α/m 定档，另报未调整 α；SE 为选定推断在真实数据上的'
                               '标准误（结果变量只经残差进入 SE）；MDE_sd = MDE / 锁定 SD，分档门槛为 0.10 / 0.25；本文件不含任何估计或 p。'
                               '五种情形的模拟 MDE（每单位误差 SD）并列，只作描述，不定档。观测精度 MDE 不等于数据独立的'
                               '事前功效分析'}
    output.write_json(f'{base}/mde.json', mde_record)
    if recompute:
        log(f'{encoder}: coverage / size simulation (design only, no outcome is read)')
        sim = roundtrip(simulation.run(panel, progress=log))
    else:
        sim = archived(encoder, 'simulation')
    sim_record = {'encoder': encoder, 'origin': origin, **sim, 'verdicts': simulation_verdicts(sim),
                  'written_before_any_estimate': True,
                  'statement': '事前规定的覆盖与尺寸模拟（只用设计：行、国家、月份、文档、权重；不读取任何结果变量），'
                               '与 MDE 一并先于任何估计确定；通过只是模拟验收标准，不构成有效性证明'}
    output.write_json(f'{base}/simulation.json', sim_record)
    return {'selection': roundtrip(sel_record), 'mde': roundtrip(mde_record), 'simulation': roundtrip(sim_record)}


def archive_matches(encoder: str, design: dict) -> dict:
    """Recomputed design records against the archived ones: {record: identical?}."""
    out = {}
    for name, key in (('inference_selection', 'selection'), ('simulation', 'simulation')):
        mine = {k: v for k, v in design[key].items()
                if k not in ('encoder', 'origin', 'verdicts', 'written_before_any_estimate', 'statement')}
        out[name] = mine == archived(encoder, name)
    return out


# --------------------------------------------------------------------------- step 3: estimates
def theta_difference(m1: dict, m0: dict, responses: list) -> dict:
    """theta_M1 - theta_M0 on the same draws (complete case over the draws where both are estimable)."""
    out = {}
    if 'theta' not in m1['estimands'] or 'theta' not in m0['estimands'] or '_valid' not in m1 or '_valid' not in m0:
        return {r: {'status': inf.UNAVAILABLE} for r in responses}
    B = m1['draws']['B']
    valid = m1['_valid']['theta'] & m0['_valid']['theta']
    eff = inf.effective(valid, B)
    for j, r in enumerate(responses):
        a, b = m1['estimands']['theta'][r], m0['estimands']['theta'][r]
        if not eff['available'] or a.get('status') != 'ok' or b.get('status') != 'ok':
            out[r] = {'status': inf.UNAVAILABLE, 'reason': eff.get('reason')}
            continue
        draws = m1['_all_estimates']['theta'][eff['index'], j] - m0['_all_estimates']['theta'][eff['index'], j]
        out[r] = {'status': 'ok', **inf.draw_inference(a['estimate'] - b['estimate'], draws)}
    return out


def wcb_block(panel: pd.DataFrame, spec: est.Spec, responses: list) -> dict:
    sub = est.select(panel, spec)
    post = est.post_indicator(sub, spec)
    design = est.build_design(sub, spec, post)
    struct = est.structural(design.X, design.names, design.groups)
    contrasts, status = est.reduce_contrasts(design, struct)
    X = design.X[:, struct['kept']]
    out = {}
    for j, r in enumerate(responses):
        pc_index = int(r[-1]) - 1
        out[r] = inf.wcb_v1(X, sub[r].to_numpy(float), sub['doc_id'].to_numpy(), contrasts, pc_index)
    return {'responses': out, 'role': 'document-cluster WCB of the main analysis, reference only; the selected '
                                      'inference decides theta and delta_UK; disagreements are reported side by side'}


def main_block(panel: pd.DataFrame, sel: dict, cache: est.DrawCache) -> dict:
    """The main models of both samples: M1 (month-block draws), the selected M1 inference, M0, M2,
    theta_M1 - theta_M0 and the document-cluster WCB of M1 and M0 (private draw arrays still attached)."""
    R = panel_mod.RESPONSES
    main = {}
    for sample in CONTROLS:
        log(f'main models {sample}')
        m1 = est.fit_spec(panel, est.main_spec(sample), R, cache, keep_draws=True)
        m0 = est.fit_spec(panel, est.main_spec(sample, 'M0'), R, cache, keep_draws=True)
        m2 = est.fit_spec(panel, est.main_spec(sample, 'M2'), R, cache)
        chosen = selection.apply(panel, sel, 'M1', sample, R, cache)
        main[sample] = {'M1': m1, SELECTED: chosen, 'M0': m0, 'M2': m2,
                        'theta_minus_theta_M0': theta_difference(m1, m0, R),
                        'wcb_v1': {'M1': wcb_block(panel, est.main_spec(sample), R),
                                   'M0': wcb_block(panel, est.main_spec(sample, 'M0'), R)}}
    return main


def run_main_only(encoder: str, panel: pd.DataFrame, panel_record: dict) -> dict:
    """The main models of one encoder (Table 9 panel B), with the archived inference selection; written to
    results/external/<encoder>/estimate/main.json."""
    sel = archived(encoder, 'inference_selection')
    main = main_block(panel, sel, est.DrawCache())
    output.write_json(f'{OUT}/{encoder}/estimate/main.json', est.strip_private(main))
    output.write_json(f'{OUT}/{encoder}/panel.json', panel_record)
    return roundtrip(est.strip_private(main))


def run_estimate(encoder: str, panel: pd.DataFrame, panel_record: dict, documents: pd.DataFrame,
                 design: dict) -> dict:
    """Every estimate of one encoder (written under results/external/<encoder>/estimate/); returns the summary."""
    base = f'{OUT}/{encoder}/estimate'
    R = panel_mod.RESPONSES
    cache = est.DrawCache()
    sel, mde, sim = design['selection'], design['mde'], design['simulation']
    output.write_json(f'{OUT}/{encoder}/panel.json', panel_record)
    calibration = {}
    for sample in CONTROLS:
        log(f'{encoder}: fake cut-off diagnostic {sample}')
        calibration[sample] = inf.fake_cutoff_diagnostic(panel, est.main_spec(sample), R, cache)
    output.write_json(f'{base}/fake_cutoffs.json', calibration)
    main = main_block(panel, sel, cache)
    output.write_json(f'{base}/main.json', est.strip_private(main))
    log(f'{encoder}: pre-trend slope diagnostic (pre-signing rows, month-block bootstrap)')
    slopes = pretrend.pretrend(panel, R, cache, simulation=sim)
    for sample in CONTROLS:
        slopes['samples'][sample]['selected_inference'] = selection.apply(panel, sel, 'slope', sample, R, cache)
    output.write_json(f'{base}/pretrend_slopes.json', est.strip_private(slopes))
    events, placebo = {}, {}
    for sample in CONTROLS:
        log(f'{encoder}: event study (with HonestDiD through R) and placebos {sample}')
        spec = est.main_spec(sample)
        sd = est.locked_sd(panel, spec, R)
        events[sample] = eventstudy.event_study(panel, spec, R, cache, sd,
                                                simulation=((sim or {}).get('samples') or {}).get(sample))
        placebo[sample] = eventstudy.placebos(panel, spec, R, cache)
    output.write_json(f'{base}/event_study.json', events)
    output.write_json(f'{base}/placebos.json', est.strip_private(placebo))
    log(f'{encoder}: sensitivity samples')
    concurrent = sensitivity.concurrent_documents(documents)
    sens = sensitivity.run(panel, R, cache, concurrent, progress=log)
    output.write_json(f'{base}/sensitivity.json', est.strip_private(sens))
    summary = estimate_summary(encoder, main, events, calibration, mde, sens, slopes, sim, placebo)
    summary['own_pca'] = (panel_record or {}).get('own_pca')
    output.write_json(f'{base}/summary.json', summary)
    return roundtrip(summary)


def estimate_summary(encoder: str, main: dict, events: dict, calibration: dict, mde: dict, sens: dict | None = None,
                     slopes: dict | None = None, sim: dict | None = None, placebo: dict | None = None) -> dict:
    """The per-encoder numbers of the report: the main-model rows (the selected inference for theta and delta_UK of
    M1), theta_M1 - theta_M0, the cross-control comparison on the same window (S_SG 2017+ vs S_CA 2017+), the pre-trend
    slope diagnostic, the descriptive event study and its sensitivity blocks (not inference), and the simulation
    judgements."""
    rows = []
    for sample in CONTROLS:
        for model in ('M1', 'M0', 'M2'):
            fit = main[sample][model]
            for name, per in fit['estimands'].items():
                for r, e in per.items():
                    row = {'encoder': encoder, 'sample': sample, 'model': model, 'estimand': name, 'response': r,
                           'status': e.get('status'), 'estimate': e.get('estimate'), 'p': e.get('p'),
                           'ci95': e.get('ci95'), 'se': e.get('se'), 'B_e': e.get('B_e'), 'failures': e.get('failures')}
                    chosen = main[sample].get(SELECTED) if model == 'M1' else None
                    if chosen is not None and name in chosen['estimands']:
                        # p and intervals of the selected candidate; the L = 6 month-block values are kept alongside
                        sel_e = chosen['estimands'][name].get(r) or {}
                        row.update(p_L6_month_block=row['p'], ci95_L6_month_block=row['ci95'],
                                   inference=chosen.get('winner') if chosen['status'] == 'selected' else DESCRIPTIVE,
                                   p=sel_e.get('p'), ci95=sel_e.get('ci95'), ci90=sel_e.get('ci90'))
                        if chosen['status'] != 'selected':
                            row['status'] = DESCRIPTIVE
                    elif model == 'M1':
                        row['inference'] = ('L = 6 month-block (descriptive estimand; the inference selection covers '
                                            'theta and delta_UK)')
                    if model == 'M1' and name in ('theta', 'delta_UK'):
                        cal = calibration[sample]['summary'].get(name, {}).get(r, {})
                        row['calibration_flag'] = cal.get('flag')
                        row['fake_cut_rejection_rate'] = cal.get('empirical_rejection_rate')
                        m = ((mde['mde'].get(sample) or {}).get('estimands', {}).get(name, {}).get(r) or {})
                        row['mde_tier'] = (m.get('tier') or {}).get('tier')
                        row['mde_member'] = m.get('member')
                    rows.append(row)

    def sens_summary(block: dict) -> dict:
        fams = {}
        for f, v in (block.get('families') or {}).items():
            j, t, h = v.get('joint') or {}, v.get('tost') or {}, v.get('honestdid') or {}
            fams[f] = {'joint': {k: j.get(k) for k in ('status', 'test', 'statistic', 'df', 'df1', 'df2', 'p', 'reason')},
                       'tost_pass': t.get('pass'), 'tost_epsilon': t.get('epsilon'),
                       'honestdid': {k: h.get(k) for k in ('status', 'breakdown', 'breakdown_status', 'reason')},
                       'simulation_size': v.get('simulation')}
        return {'label': block.get('label'), 'assumption': block.get('assumption'), 'valid_inference': False,
                'families': fams}

    def descriptive(pc: dict) -> dict:
        return {k: (v or {}).get('estimate') for k, v in (pc.get('coefficients') or {}).items()
                if k.split('@')[0] in ('theta', 'delta_UK', 'delta_US')}
    for sample in CONTROLS:
        for r, e in main[sample]['theta_minus_theta_M0'].items():
            rows.append({'encoder': encoder, 'sample': sample, 'model': 'M1-M0', 'estimand': 'theta_minus_theta_M0',
                         'response': r, 'status': e.get('status'), 'estimate': e.get('estimate'), 'p': e.get('p'),
                         'ci95': e.get('ci95'), 'se': e.get('se'), 'B_e': e.get('B_e')})
    same_window = {}
    if sens:
        pairs = {'S_SG 2017+ (sensitivity shrink_2017)': (sens.get('SG') or {}).get('shrink_2017'),
                 'S_CA 2017+ (main)': main['CA']['M1']}
        for label, fit in pairs.items():
            if fit is None:
                continue
            same_window[label] = {name: {k: (fit['estimands'][name].get('ref_PC1') or {}).get(k)
                                         for k in ('status', 'estimate', 'p', 'ci95')}
                                  for name in ('theta', 'delta_UK', 'delta_US', 'delta_AU')}
    pre = {}
    for sample, block in ((slopes or {}).get('samples') or {}).items():
        def fit_summary(fit):
            return {name: {k: (per.get('ref_PC1') or {}).get(k) for k in ('status', 'estimate', 'p', 'ci95', 'se', 'B_e',
                                                                           'failures', 'reason')}
                    for name, per in fit['estimands'].items()}
        pre[sample] = {'main': fit_summary(block['main']), 'block_length': block['main']['draws']['block_length'],
                       'sensitivity': {k: fit_summary(v) for k, v in block['sensitivity'].items()},
                       'simulation': block.get('simulation'),
                       'selected_inference': {'status': (block.get('selected_inference') or {}).get('status'),
                                              'winner': (block.get('selected_inference') or {}).get('winner'),
                                              'estimands': {nm: (per or {}).get('ref_PC1') for nm, per in
                                                            ((block.get('selected_inference') or {}).get('estimands')
                                                             or {}).items()}}}
    sim_labels = {}
    for sample, block in ((sim or {}).get('samples') or {}).items():
        sim_labels[sample] = {name: {'by_item': proc.get('by_item'), 'by_joint': proc.get('by_joint'),
                                     'verdicts': proc.get('verdicts')}
                              for name, proc in (block.get('procedures') or {}).items()}

    def fit_summary_all(fit):
        return {name: {k: (per.get('ref_PC1') or {}).get(k) for k in ('status', 'estimate', 'p', 'ci95', 'reason')}
                for name, per in (fit.get('estimands') or {}).items()}
    wcb, side = {}, {}
    for sample in CONTROLS:
        wcb[sample] = {}
        for model in ('M1', 'M0'):
            resp = (((main[sample].get('wcb_v1') or {}).get(model) or {}).get('responses') or {}).get('ref_PC1') or {}
            wcb[sample][model] = {name: {k: v.get(k) for k in ('estimate', 'se', 'p_v1', 'p_plus_one')}
                                  for name, v in (resp.get('contrasts') or {}).items()}
        for name in ('theta', 'delta_UK', 'delta_US', 'delta_AU'):
            row = next((x for x in rows if x['sample'] == sample and x['model'] == 'M1' and x['estimand'] == name
                        and x['response'] == 'ref_PC1'), {})
            w = wcb[sample]['M1'].get(name) or {}
            p_dec, p_w = row.get('p'), w.get('p_plus_one')
            side[f'{sample}:{name}'] = {
                'estimate': row.get('estimate'), 'deciding_inference': row.get('inference'), 'p_deciding': p_dec,
                'ci95_deciding': row.get('ci95'), 'p_L6_month_block': row.get('p_L6_month_block', row.get('p')),
                'wcb_estimate': w.get('estimate'), 'p_wcb_v1': p_w, 'se_wcb_v1': w.get('se'),
                'conflict_at_05': (None if p_dec is None or p_w is None else bool((p_dec < 0.05) != (p_w < 0.05))),
                'rule': 'the WCB is a reference; the deciding inference is the selected one (theta, delta_UK) or the '
                        'descriptive M1 rows; a disagreement is reported, never resolved in favour of either'}
    placebo_summary = {s: {cut: {'origin': fit.get('origin'), **fit_summary_all(fit)} for cut, fit in blk.items()}
                     for s, blk in (placebo or {}).items()}
    sens_summary_all = {s: {item: {'description': fit.get('description'), **fit_summary_all(fit)}
                          for item, fit in blk.items() if isinstance(fit, dict) and 'estimands' in fit}
                      for s, blk in (sens or {}).items() if isinstance(blk, dict)}
    return {'method_note': {'deciding_inference': selection.GUARANTEE_SCOPE,
                            'scope': 'M1 theta / delta_UK and the pre-trend slope theta_slope / slope_UK where the '
                                     'inference selection chose C8; windows in the fallback are descriptive (they enter the '
                                     'Holm family with p = 1)'},
            'rows': rows, 'cross_control_same_window': same_window,
            'wcb_v1': wcb, 'wcb_vs_deciding': side, 'placebos': placebo_summary, 'sensitivity': sens_summary_all,
            'pretrend_slopes': {'statement': (slopes or {}).get('statement'), 'samples': pre},
            'events_descriptive': {s: {r: descriptive(pc) for r, pc in ev['PCs'].items()} for s, ev in events.items()},
            'events_sensitivity': {s: {r: {name: sens_summary(b) for name, b in (pc.get('sensitivity') or {}).items()}
                                       for r, pc in ev['PCs'].items()} for s, ev in events.items()},
            'event_reading': {s: (ev.get('inference') or {}).get('reading') for s, ev in events.items()},
            'event_block_draw_absence': {s: (ev.get('block_draw_period_absence') or {}).get('shares')
                                         for s, ev in events.items()},
            'simulation': sim_labels,
            'simulation_descriptive': {s: b.get('descriptive') for s, b in ((sim or {}).get('samples') or {}).items()}}


# --------------------------------------------------------------------------- Table 9
INFERENCE_TEXT = {'C1': 'month-block bootstrap (L = 12)', 'C2': 'month-block bootstrap (L = 24)',
                  'C3': 'studentized month-block bootstrap (L = 12)', 'C4': 'CR2 cluster-robust inference'}
for _c, _base in (('C5', 'C1'), ('C6', 'C2'), ('C7', 'C3'), ('C8', 'C4')):
    INFERENCE_TEXT[_c] = f'simulation-calibrated {INFERENCE_TEXT[_base]}'
SAMPLE_NAME = {'SG': 'Singapore', 'CA': 'Canada'}


def table9(summary: dict, mde_record: dict, mains: dict) -> pd.DataFrame:
    """Panel A: the baseline encoder, both controls, theta / delta_UK (selected inference, with the MDE of the
    selected inference) and delta_US (point estimate). Panel B: the Singapore theta point estimate of every
    alternative encoder."""
    rows = []
    mde = mde_record['mde']
    for sample in CONTROLS:
        for name in ('theta', 'delta_UK', 'delta_US'):
            r = next(x for x in summary['rows'] if x['sample'] == sample and x['model'] == 'M1'
                     and x['estimand'] == name and x['response'] == 'ref_PC1')
            inferred = name in ('theta', 'delta_UK')
            m = ((mde.get(sample) or {}).get('estimands', {}).get(name, {}).get('ref_PC1') or {})
            rows.append({'panel': 'A', 'encoder': data.DISPLAY[data.BASELINE], 'control': SAMPLE_NAME[sample],
                         'estimand': name, 'estimate': r['estimate'],
                         'lo95': r['ci95'][0] if inferred and r.get('ci95') else None,
                         'hi95': r['ci95'][1] if inferred and r.get('ci95') else None,
                         'p': r['p'] if inferred else None,
                         'inference': INFERENCE_TEXT.get(r.get('inference'), r.get('inference')) if inferred
                         else 'point estimate only',
                         'mde_sd': m.get('mde_sd') if inferred else None,
                         'mde_tier': (m.get('tier') or {}).get('tier') if inferred else None,
                         'mde_tier_text': (m.get('tier') or {}).get('text') if inferred else None})
    for encoder, main in mains.items():
        e = main['SG']['M1']['estimands']['theta']['ref_PC1']
        rows.append({'panel': 'B', 'encoder': data.DISPLAY[encoder], 'control': SAMPLE_NAME['SG'], 'estimand': 'theta',
                     'estimate': e['estimate'], 'lo95': None, 'hi95': None, 'p': None,
                     'inference': 'point estimate only', 'mde_sd': None, 'mde_tier': None, 'mde_tier_text': None})
    return pd.DataFrame(rows)
