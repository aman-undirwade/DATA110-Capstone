# Notebooks

The repository now contains the executable capstone notebook:

`Aman_Undirwade_DATA110_Capstone_Predictive_Maintenance.ipynb`

It is aligned with the Aircraft Engine Predictive Maintenance report and presentation and follows this order:

1. Data loading and schema check
2. Exploratory analysis
3. RUL and warning-label construction
4. Engine-level validation
5. Feature engineering
6. Five-model comparison across 10/20/30/50-cycle horizons
7. Threshold selection and error analysis
8. Official FD004 test evaluation
9. SHAP explainability
10. Reproducible result export

The notebook imports the reusable modelling functions from `src/`, so the notebook and Python implementation share the same core logic rather than maintaining two independent implementations.

## Data requirement

The NASA C-MAPSS FD004 raw benchmark is not committed to this repository. Before executing the notebook, place:

- `train_FD004.txt`
- `test_FD004.txt`
- `RUL_FD004.txt`

inside `data/raw/`.

The `.gitignore` intentionally excludes `/data/raw/`, generated experiment outputs, and model binaries.
