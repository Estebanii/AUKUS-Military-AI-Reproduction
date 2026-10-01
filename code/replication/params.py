"""Fixed parameters of the main analysis (baseline encoder and the four alternative encoders).

Seeds and resampling counts come from :mod:`replication.seeds`; nothing here is random. :func:`params` returns a deep
copy of one section, so a caller can never change the parameters of another step.
"""
from __future__ import annotations

import copy

from . import seeds

# GPT-2 (Radford et al. 2019) provides the label space of the embedding regression: the input embeddings of the anchor
# words (labels of the A matrix) and the vocabulary of the neighbour-word analysis. The files are pinned by revision
# and sha256 and verified before use.
GPT2 = {
    'model_id': 'gpt2',
    'revision': '607a30d783dfa663caf39e06633721c8d4cfcd7e',
    'files_sha256': {
        'config.json': '0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4',
        'model.safetensors': '248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707',
        'merges.txt': '1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5',
        'vocab.json': '196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783',
        'tokenizer.json': '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6',
        'tokenizer_config.json': '5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f',
    },
}

PARAMS = {
    # labels of the A matrix: mean of the GPT-2 input embeddings of each anchor word's sub-tokens (float32)
    'labels': {'source': 'gpt2 wte', 'aggregation': 'mean of bare-word sub-tokens', 'add_special_tokens': False,
               'dtype': 'float32', 'key': 'lower-case word', 'gpt2': GPT2},
    # the A matrix: ridge regression of the centred labels on the whitened, centred encoder states with a Procrustes
    # prior (Khodak et al. 2018; Rodriguez, Spirling and Stewart 2023). The prior is the rank-r part of the Procrustes
    # solution, r = #{singular values > 1e-8 x the largest}; every step after the float32 encoder states is float64.
    'a_matrix': {'lambda': 0.1, 'whiten': True, 'whiten_eps': 1e-6, 'procrustes_prior': True,
                 'rule': 'rank_truncated', 'a0_rank_rel_tol': 1e-8, 'y_formula': '(U - U_mean) @ A.T'},
    # principal components: full SVD, deterministic
    'pca': {'n_components': 88, 'svd_solver': 'full', 'random_state': None, 'k99_threshold': 0.99},
    # H1 MANOVA (Wilks' lambda, Rao's F): PCA-88 of all rows; test 6 = pre-signing US vs UK from 2014
    'manova': {'n_pc': 88, 'test6': {'countries': ['US', 'UK'], 'min_year': 2014, 'period': 'pre'}},
    # H2 wild cluster bootstrap (Cameron, Gelbach and Miller 2008)
    'wcb': {'n_bootstrap': seeds.WCB_DRAWS, 'seed_base': seeds.WCB_SEED_BASE,
            'seeds': [seeds.WCB_SEED_BASE + i for i in range(3)], 'weights': 'rademacher',
            'residuals': 'unrestricted', 'p_value': 'centred tail: mean(|b*-b| >= |b|)', 'trend_origin_year': 2014,
            'post_rule': 'year_month >= 2021-09 (post_aukus)', 'cluster': 'doc_id (numpy.unique order)',
            'interval': 'b +/- quantile(|b*-b|, 0.95, method=linear) (not multiplicity-adjusted)',
            'interval_level': 0.95, 'quantile_method': 'linear'},
    # robustness models (year fixed effects, 2017+, placebos) and the event-study Wald of the robustness block
    'did_robustness': {'ref_year_fe': 2021, 'restricted_min_year': 2017, 'placebo_years': [2020, 2021],
                       'event_study': {'min_year': 2017, 'max_year': 2024, 'base_year': 2021,
                                       'seed': seeds.EVENT_STUDY_WALD_SEED}},
    # parallel-trends event study (US and UK, 2014-2024, base year 2021): joint Wald on the pre-signing UK x year terms
    'parallel_trends': {'countries': ['US', 'UK'], 'min_year': 2014, 'max_year': 2024, 'base_year': 2021,
                        'wald': 'full bootstrap covariance of the pre-2021 UK x year coefficients'},
    # H3 neighbour words
    'h3': {'top_k': 30, 'top_n_compare': 15, 'filter': 'G-prefix, alphabetic, len>1, lower-case dedup',
           'periods': ['pre_aukus', 'post_aukus', 'full_period']},
    # alternative encoders: own-PC1 orientation against the reference PC1 loading
    'orientation': {'primary_threshold': 0.30, 'descriptive_thresholds': [0.50, 0.70]},
    # alternative encoders: Holm families over the four encoders
    'families': {'same_procedure': 'alternative encoders, own-PCA PC1 UK x Post (oriented by the reference PC1), '
                                   'Holm over 4',
                 'same_axis': 'alternative encoders, reference-axis PC1 UK x Post, Holm over 4',
                 'alpha': 0.05},
    # alternative encoders: Monte Carlo precision of the bootstrap p values and the descriptive re-run
    'cross_encoder_reporting': {'mc_B': seeds.WCB_DRAWS, 'se_multiplier': 2,
                                'boundary_flag': '边界：对 bootstrap 抽样敏感', 'p_zero_display': '< 0.001',
                                'rerun_seed': seeds.RERUN_SEED, 'rerun_B': seeds.RERUN_DRAWS,
                                'rerun_p': '(k + 1) / (B + 1)', 'v1_pc1_uk_x_post': 0.033255226,
                                'labels': {'holm_not_rejected': '未达到预定复现判据',
                                           'consistent_but_imprecise': '点估计一致、精度不足以达到预定判据'}},
}


def params(section: str | None = None) -> dict:
    """A deep copy of the parameters (or one section)."""
    return copy.deepcopy(PARAMS if section is None else PARAMS[section])
