"""Every random seed and resampling count of the replication, in one place.

All drivers and modules read their seeds from here (``replication.params`` refers to these names); no seed is set
anywhere else in ``code/`` or ``scripts/``. README section 6 ("随机种子") lists, for every stochastic step, the seed,
the number of draws and which output it affects.

Rerunning in the pinned environment (``environment/``) with the pinned BLAS thread count reproduces every recomputed
number bit for bit.
"""

# --------------------------------------------------------------------------- main analysis (baseline encoder)
# Wild cluster bootstrap for the subgroup DiD (Table 4) and its robustness models (Tables 5-6): Rademacher weights by
# document, one generator per principal component with seed WCB_SEED_BASE + (PC index 0, 1, 2) = 42, 43, 44.
WCB_SEED_BASE = 42
WCB_DRAWS = 1000
# The event-study Wald test of the robustness block (three countries, 2017-2024) uses one generator, seed 42, for
# every component (as in the original specification).
EVENT_STUDY_WALD_SEED = 42

# --------------------------------------------------------------------------- alternative encoders (Table 8)
# Descriptive re-run of the PC1 UK x Post bootstrap for every alternative encoder (and the baseline on the common
# rows), p = (k + 1) / (B + 1); reported beside the primary B = 1,000 result, never replacing it.
RERUN_SEED = 20260929
RERUN_DRAWS = 9999

# --------------------------------------------------------------------------- distance change (Table 7)
# Pivot bootstrap of the squared-distance change: documents resampled with replacement within each country x period
# cell, B = 2,000, numpy.random.default_rng(DISTANCE_SEED); fixed draw order (see replication.distance).
DISTANCE_SEED = 20260929
DISTANCE_DRAWS = 2000

# --------------------------------------------------------------------------- external controls (Table 9)
# One base seed for the external-control analysis (see replication.external.params): the month-block bootstrap
# (B = 9,999, PCG64), the development draws of the calibration / selection simulation of the cluster-robust inference
# (2,000 replications per scenario) and the coverage / size simulation (2,000 replications per scenario). The selected
# inference is re-verified with independent draws (EXTERNAL_VERIFICATION_SEED). The reference wild cluster bootstrap
# of the external-control models uses WCB_SEED_BASE + PC index with B = 1,000.
EXTERNAL_SEED = 20260928
EXTERNAL_VERIFICATION_SEED = 20260929
EXTERNAL_BLOCK_BOOTSTRAP_DRAWS = 9999
# At most this share of the B draws may fail (target contrast not estimable); failed draws are replaced by further
# draws from the same generator, of which R = 2 floor(share B) + 1 are drawn after the B main draws (199 at B = 9,999).
EXTERNAL_MAX_FAILURE_SHARE = 0.01
EXTERNAL_SIMULATION_REPLICATIONS = 2000
# The descriptive HonestDiD in-restriction check of the coverage / size simulation uses the first replications of its
# iid scenario (Singapore window).
EXTERNAL_HONESTDID_CHECK_REPLICATIONS = 50

# --------------------------------------------------------------------------- figures
# Figure 1 (PCA scatter) and Figure 2 (t-SNE): the seeds of the original figure script.
FIGURE_PCA_RANDOM_STATE = 42
TSNE_RANDOM_STATE = 42

# --------------------------------------------------------------------------- upstream data preparation (not re-executed)
# The anchor sample of the A matrix (49,999 anchor occurrences, stratified by country) was drawn once with
# pandas.DataFrame.sample(random_state=ANCHOR_SAMPLE_RANDOM_STATE); the drawn rows are part of the data bundle.
ANCHOR_SAMPLE_RANDOM_STATE = 42

# --------------------------------------------------------------------------- calibration (supplementary)
# Synthetic calibration of the distance interval (supplementary/distance_calibration): data seeds
# numpy.random.SeedSequence([CALIBRATION_DATA_SEED, setting index]); its bootstrap uses DISTANCE_SEED.
CALIBRATION_DATA_SEED = 20260930
CALIBRATION_REPS = 500                    # replications per setting (default of --reps)
# Second entropy word of the fixed random direction of the UK mean: numpy.random.default_rng([DATA_SEED, KEY]).
CALIBRATION_DIRECTION_KEY = 999
# Setting indices: the delta_sq = 0 settings are 0, 1, ...; the non-zero settings start at the first offset and the
# stress settings at the second.
CALIBRATION_NONZERO_INDEX_OFFSET = 100
CALIBRATION_STRESS_INDEX_OFFSET = 200
