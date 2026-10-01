"""Fixed parameters of the external-control analysis (Singapore and Canada as non-member controls; Table 9).

Seeds and resampling counts come from :mod:`replication.seeds`. :func:`external_params` returns a deep copy of one section.
"""
from __future__ import annotations

import copy

from .. import seeds

SEED = seeds.EXTERNAL_SEED
CONTROLS = ('SG', 'CA')
MEMBERS = ('US', 'UK', 'AU')

EXTERNAL_PARAMS = {
    # the study window of every country ends 2025-11; 2021-09 is wholly post (month-level coding, as in the main analysis)
    'extract': {'cut_last_month': '2025-11', 'post_first_month': '2021-09'},
    # windows of the two controls and of the sensitivity samples
    'windows': {'SG': {'start': '2014-01', 'end': '2025-11'}, 'CA': {'start': '2017-01', 'end': '2025-11'},
                'shrink_2017': {'start': '2017-01', 'end': '2025-11'},
                'common_2017_09': {'start': '2017-09', 'end': '2025-11'},
                'SG_to_2023_12': {'start': '2014-01', 'end': '2023-12'},
                'pooled': {'start': '2017-01', 'end': '2025-11'}},
    'trend_origin_year': 2014,
    # fixed column priority of the pivoted QR (only columns linearly dependent on higher-priority columns are dropped;
    # an all-zero column is dropped as well); term fixed effects come last
    'column_priority': ['constant', 'country levels', 'time terms', 'Post main effect', 'interactions',
                        'event-period terms', 'term FE'],
    'qr_rel_tol': 1e-9,
    # circular month-block bootstrap
    'bootstrap': {'B': seeds.EXTERNAL_BLOCK_BOOTSTRAP_DRAWS, 'seed': SEED, 'block_length': 6,
                  'block_length_sensitivity': [3, 12],
                  'scheme': 'circular within the pre and the post layer (Politis-Romano); block starts uniform over the '
                            'layer; one numpy Generator(PCG64(seed)): pre starts (B, blocks) then post starts',
                  'unit': 'calendar month of the window (2021-09 whole in post)',
                  'max_failure_share': seeds.EXTERNAL_MAX_FAILURE_SHARE, 'alpha': 0.05, 'p': '(1 + #{|t*-t|>=|t|}) / (B_e + 1)',
                  'interval': 't +/- a_(B_e - r), r = ceil(alpha (B_e + 1)) - 2 (order statistic of |t*-t|)'},
    # fake cut-off diagnostic
    'fake_cutoffs': {'first': '2016-01', 'last': '2019-12', 'min_months_after': 18, 'alpha': 0.05, 'flag_above': 0.10},
    # the document-cluster wild cluster bootstrap of the main analysis, reported beside (historical reference)
    'wcb_v1': {'B': seeds.WCB_DRAWS, 'seed_base': seeds.WCB_SEED_BASE, 'weights': 'Rademacher', 'cluster': 'doc_id'},
    # minimum detectable effect
    'mde': {'z_power': 0.8416212335729143, 'power': 0.80, 'alpha': 0.05, 'tiers_sd': [0.10, 0.25], 'total_m': 10,
            'v1_reference': {'pc1_uk_x_post': 0.033255226, 'pre_sd': 0.342361443, 'sd_units': 0.097134847},
            'sd_population': 'estimation sample (members + control), pre period, occurrence equal weight, that axis',
            'c': 'order statistic of |t* - mean(t*)| at the family level (dimensionless, / SE)'},
    # test families of the external-control analysis
    'families': {'baseline_family': {'encoders': ['deberta-v3-base'], 'estimands': ['theta', 'delta_UK'], 'm': 2},
                 'alternatives_theta': {'encoders': ['ModernBERT-large', 'roberta-large', 'ettin-encoder-1b',
                                                     'deberta-v2-xlarge'], 'estimands': ['theta'], 'm': 4},
                 'alternatives_delta_UK': {'encoders': ['ModernBERT-large', 'roberta-large', 'ettin-encoder-1b',
                                                        'deberta-v2-xlarge'], 'estimands': ['delta_UK'], 'm': 4},
                 'member_spec': {'sample': 'SG', 'model': 'M1', 'axis': 'same_axis', 'pc': 'PC1'},
                 'alpha': 0.05},
    # event study (period coding 2014..2020, 2021a, 2021b, 2022..2025; reference 2020)
    'event_study': {'reference': '2020', 'pre_sets': {'SG': ['2014', '2015', '2016', '2017', '2018', '2019', '2021a'],
                                                      'CA': ['2017', '2018', '2019', '2021a']},
                    'post': ['2021b', '2022', '2023', '2024', '2025'], 'tost_sd': 0.10,
                    # the saturated event study is descriptive; its Wald, TOST and HonestDiD appear only as
                    # sensitivity under the estimators below
                    'inference': {
                        'decision': 'descriptive event study',
                        'reading': 'event-study coefficients are descriptive; Wald, TOST and HonestDiD on the saturated '
                                   'event study are not valid inference and are reported only as sensitivity under the '
                                   'estimators of sensitivity_estimators, each labelled with its assumption',
                        'why': 'the design is saturated in country x event period, so each coefficient\'s monthly scores '
                               'sum to zero within its period and the monthly-score HAC is biased towards zero '
                               '(balanced iid: E(V)/V = 1 - T^-2 sum_{t,u in tau} k_b(t-u); b = 6: ~0.58 / 0.49 / 0.30)',
                        'sampling_failure_gate': 'none (the month-block draws are not used for the event study)',
                        'singular': "a coefficient variance <= 0 or <= 1e-12 x c' Q^-1 c Var_w(y); a joint covariance "
                                    "with min eigenvalue <= 1e-12 x max or max <= 1e-12 x the largest c' Q^-1 c Var_w(y): "
                                    '"不可用"'},
                    'sensitivity_estimators': [
                        {'name': 'hac_bartlett_iid_exact', 'bandwidth': 6},
                        {'name': 'cr2_month'}]},
    # the low-dimensional pre-trend diagnostic (inference: the M1 synchronised month-block bootstrap)
    'pretrend': {'decision': 'pre-trend slope diagnostic', 'model': 'slope',
                 'rows': 'pre-signing rows only (ym <= 2021-08), members + control, the sample main window',
                 'trend': 'linear calendar-month trend (months since 2014-01) / 12: slopes per year',
                 'estimands': ['theta_slope', 'slope_UK', 'slope_US', 'slope_AU'],
                 'estimand_text': {'slope_g': "member g's linear pre-signing slope minus the control's",
                                   'theta_slope': 'slope_UK - slope_US'},
                 'block_length': 6, 'block_length_sensitivity': [3, 12],
                 'layers': 'one layer: the pre-signing calendar months of the window (circular blocks, all countries '
                           'resampled together)',
                 'statement': '只诊断线性偏离，不能代替逐期检验，也不能证明平行趋势成立（Roth 2022）'},
    # the result-blind selection of the M1 inference and of the pre-trend slope test inference
    'inference_selection': {
        'decision': 'inference selection',
        'procedures': {'M1': {'estimands': ['theta', 'delta_UK'], 'report_also': ['delta_US', 'delta_AU']},
                       'slope': {'estimands': ['theta_slope', 'slope_UK'], 'report_also': ['slope_US', 'slope_AU']}},
        'candidates': {'C1': {'kind': 'block', 'block_length': 12},
                       'C2': {'kind': 'block', 'block_length': 24},
                       'C3': {'kind': 'block_studentized', 'block_length': 12,
                              'se': 'month-clustered CR0 re-estimated in every draw, each drawn month copy a cluster'},
                       'C4': {'kind': 'cr2', 'single': 'Satterthwaite (Bell-McCaffrey / Imbens-Kolesar) t',
                              'joint': 'HTZ'},
                       'C5': {'kind': 'calibrated', 'base': 'C1'}, 'C6': {'kind': 'calibrated', 'base': 'C2'},
                       'C7': {'kind': 'calibrated', 'base': 'C3'}, 'C8': {'kind': 'calibrated', 'base': 'C4'}},
        'calibration': 'S = |estimate| / nominal 95% half-width of the base candidate; the critical value at level a is '
                       'the maximum over the five development scenarios of the (1 - a) quantile of S (method higher); '
                       'test: S > c*_a; interval: estimate +/- c*_a x nominal 95% half-width; p = the maximum over the '
                       'scenarios of (1 + #{S_s >= S_obs}) / (R + 1) (development null draws)',
        'development_seed': SEED, 'verification_seed': seeds.EXTERNAL_VERIFICATION_SEED,
        'replications': seeds.EXTERNAL_SIMULATION_REPLICATIONS,
        'levels': [0.10, 0.05, 0.025, 0.0125, 0.005],
        'admission': 'every scenario x estimand in scope: coverage95, coverage90 and size05 all 通过 (thresholds of '
                     'the simulation section, one-sided 95% Clopper-Pearson bounds)',
        'selection': 'among admitted candidates the smallest worst-case size05 upper bound (over scenarios and estimands); '
                     'ties: the wider mean standardised 95% interval (mean half-width / simulated SD of the estimate, '
                     'averaged over scenarios and estimands); then the candidate number',
        'verification': 'the winner rerun with the verification seed, 2,000 replications per scenario (calibrated '
                        'candidates keep the development critical values); every scenario x estimand x metric 通过, '
                        'else fallback',
        'fallback': 'descriptive (enters the Holm family with p = 1); no further switching',
        'holm_levels': 'simulated sizes at alpha / j, j = 1..10 (verification replications)',
        'mde': {'rule': 'MDE = (c + z_.80) x SE with c the critical value of the test actually used (observed '
                        'precision): per window, c = the selected M1 inference\'s critical value on the SE scale at the '
                        'Holm first-step level alpha / m (members; other rows alpha), the unadjusted alpha alongside -- '
                        'calibrated candidates c = c*_a x nominal 95% half-width / SE (C8: c*_a x the Satterthwaite '
                        't_.975 of the contrast); SE = the selected candidate\'s SE on the real data (C8: month-clustered '
                        'CR2); MDE_sd = MDE / the locked pre-period SD; tiers 0.10 / 0.25; a window in the selection '
                        'fallback has no MDE (descriptive)',
                'formula': 'MDE_SD = (c*_{alpha/m} x t_{.975,nu} + z_.80) x SE_CR2 / SD_locked (C8: c* the calibrated '
                           'multiplier of the window, nu the Satterthwaite (BM/IK) df of the contrast)',
                'tier_rule': 'observed_precision', 'power': 0.80,
                'scenario_mde': 'descriptive only, never sets a tier: per scenario the simulated-power MDE per unit '
                                'error SD of the verified winner (a true effect delta shifts the estimate and leaves every '
                                'half-width unchanged; the smallest delta >= 0 with power >= 0.80 over the verification '
                                'replications), and its conversion x the M1 residual SD / the locked SD'}},
    # coverage / size simulation of the procedures actually used (design only; no outcome is read)
    'simulation': {
        'decision': 'coverage / size simulation', 'replications': seeds.EXTERNAL_SIMULATION_REPLICATIONS,
        'seed': SEED, 'batch': 200,
        'truth': 'Y = e: every coefficient, slope and event contrast is 0; the real design of each sample (rows, '
                 'countries, calendar months, documents, weights); no outcome is read',
        'components': 'idiosyncratic N(0,1) per row; document N(0,1) per document; country_month stationary AR(1) per '
                      'country over the calendar months of the window; common_month one stationary AR(1) over the '
                      'calendar months; each scaled by the square root of its variance share',
        'scenarios': {'iid': {'idiosyncratic': 1.0},
                      'document': {'idiosyncratic': 0.7, 'document': 0.3},
                      'country_month_ar05': {'idiosyncratic': 0.6, 'document': 0.3, 'country_month': 0.1, 'rho': 0.5},
                      'country_month_ar08': {'idiosyncratic': 0.6, 'document': 0.3, 'country_month': 0.1, 'rho': 0.8},
                      'common_month_ar05': {'idiosyncratic': 0.6, 'document': 0.3, 'common_month': 0.1, 'rho': 0.5}},
        'procedures': ['M1', 'slope', 'slope_L3', 'slope_L12', 'es:hac_bartlett_iid_exact', 'es:cr2_month'],
        'thresholds': {'coverage95': 0.93, 'coverage90': 0.88, 'size05': 0.06},
        'judgement': 'one-sided 95% Clopper-Pearson bounds: coverage 失败 if the upper bound < threshold, 通过 if the lower '
                     'bound >= threshold, else 未定; size 失败 if the lower bound > .06, 通过 if the upper bound <= .06, '
                     'else 未定; every scenario reported; 通过 is a simulation acceptance criterion, not a proof of validity',
        'tost_boundary': {'epsilon_sd': 0.10, 'deviation': 'UK rows of 2021a shifted by +epsilon (delta_UK@2021a = '
                                                           'theta@2021a = epsilon; in the column space)',
                          'families': ['theta', 'delta_UK'], 'reading': 'descriptive: the rate of a TOST pass with one '
                                                                        'pre coefficient at the equivalence boundary'},
        'honestdid_in_restriction': {'sample': 'SG', 'family': 'delta_UK', 'scenario': 'iid',
                                     'replications': seeds.EXTERNAL_HONESTDID_CHECK_REPLICATIONS,
                                     'Mbar': 1.0, 'd_sd': 0.10,
                                     'deviation': 'UK rows: 2021a +d; post period k = 1..5 (2021b..2025) +d(1 + k Mbar); '
                                                  'rebased to 2021a the pre changes are 0 except the last (d) and every '
                                                  'post change is Mbar d (on the Delta^RM(Mbar) boundary); treatment 0',
                                     'reading': 'descriptive: the rate at which the HonestDiD robust interval at Mbar '
                                                'contains the true target 0'}},
    'honestdid': {'Mbar': [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], 'alpha': 0.05, 'target': 'average',
                  'grid_points': 1000, 'bisection_tol': 0.01, 'max_doublings': 10, 'estimands': ['theta', 'delta_UK'],
                  'responses': ['ref_PC1', 'own_PC1']},       # PC1 primary; PC2/PC3 descriptive without HonestDiD
    'placebo': {'month_cuts': ['2019-09', '2020-09'], 'v1_year_starts': [2020, 2021]},
    'sensitivity': {'donut': ['2021-09', '2021-12'], 'day_cut': '2021-09-16', 'genres': ['news', 'news release'],
                    'concurrent_sg': {'start': '2021-08-15', 'end': '2021-09-30', 'pattern': r'cyber',
                                      'fields': ['title', 'content']},
                    'cyber_terms': ['cyber', 'cyber capabilities']},
}


def external_params(section: str | None = None):
    """A deep copy of the parameters (or one section)."""
    return copy.deepcopy(EXTERNAL_PARAMS if section is None else EXTERNAL_PARAMS[section])
