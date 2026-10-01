"""Every random seed and resampling count of the replication, in one place."""

# --------------------------------------------------------------------------- main analysis (baseline encoder)
# wild cluster bootstrap of the main model and its robustness models: seed 42 + PC index
WCB_SEED_BASE = 42
WCB_DRAWS = 1000
# event-study Wald test of the robustness block: one seed for every component
EVENT_STUDY_WALD_SEED = 42

# --------------------------------------------------------------------------- alternative encoders (Table 8)
# descriptive B = 9,999 re-run of the PC1 UK x Post bootstrap (alternative encoders)
RERUN_SEED = 20260929
RERUN_DRAWS = 9999

# --------------------------------------------------------------------------- distance change (Table 7)
# pivot bootstrap of the squared-distance change (documents within country x period cells)
DISTANCE_SEED = 20260929
DISTANCE_DRAWS = 2000

# --------------------------------------------------------------------------- external controls (Table 9)
# month-block bootstrap, inference selection and coverage / size simulations of the external-control analysis
EXTERNAL_SEED = 20260928
EXTERNAL_VERIFICATION_SEED = 20260929
EXTERNAL_BLOCK_BOOTSTRAP_DRAWS = 9999
# share of draws that may fail; failed draws are replaced from the same generator
EXTERNAL_MAX_FAILURE_SHARE = 0.01
EXTERNAL_SIMULATION_REPLICATIONS = 2000
# HonestDiD in-restriction check: replications of the iid scenario used
EXTERNAL_HONESTDID_CHECK_REPLICATIONS = 50

# --------------------------------------------------------------------------- figures
# seeds of the original figure script
FIGURE_PCA_RANDOM_STATE = 42
TSNE_RANDOM_STATE = 42

# --------------------------------------------------------------------------- upstream data preparation (not re-executed)
# the anchor sample of the A matrix was drawn once with pandas sample(random_state=...)
ANCHOR_SAMPLE_RANDOM_STATE = 42

# --------------------------------------------------------------------------- calibration (supplementary)
# synthetic calibration of the distance interval (supplementary)
CALIBRATION_DATA_SEED = 20260930
CALIBRATION_REPS = 500                    # replications per setting (default of --reps)
# second entropy word of the random direction of the UK mean
CALIBRATION_DIRECTION_KEY = 999
# setting index offsets of the non-zero and stress settings
CALIBRATION_NONZERO_INDEX_OFFSET = 100
CALIBRATION_STRESS_INDEX_OFFSET = 200
