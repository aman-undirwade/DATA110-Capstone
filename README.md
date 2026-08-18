# DATA110 Capstone — Aircraft Engine Predictive Maintenance

## Student
**Aman Undirwade**

## Course Portfolio
This repository is organized as the DATA110 machine-learning portfolio/capstone repository. It contains the aircraft-engine predictive-maintenance capstone plus course lab/practice implementations.

## Capstone Project
The capstone studies early aircraft-engine failure detection using NASA C-MAPSS FD004. It compares Logistic Regression, Gaussian Naive Bayes, KNN, Decision Tree and Random Forest, uses engine-aware validation, evaluates warning horizons, selects a cost-sensitive operating point, evaluates the official test engines, and documents SHAP-based interpretation.

## Executable Capstone Notebook
The main presentation/viva implementation is:

`notebooks/Aman_Undirwade_DATA110_Capstone_Predictive_Maintenance.ipynb`

The notebook is designed to be opened in Jupyter Notebook, JupyterLab, VS Code or Google Colab after the FD004 raw files are placed in `data/raw/`. It imports the reusable implementation from `src/` and walks through the complete ML pipeline step by step.

### Notebook workflow
1. Load and inspect NASA C-MAPSS FD004
2. Construct RUL for training and official test trajectories
3. Build 10/20/30/50-cycle warning targets
4. Perform engine-level validation to reduce trajectory leakage
5. Compare five DATA110 classical ML algorithms
6. Select the 50-cycle Random Forest configuration
7. Select the probability threshold using an illustrative 5:1 missed-failure cost
8. Evaluate the official test engines
9. Perform error analysis
10. Explain the final model with SHAP
11. Export reproducible result tables

## Repository Structure

```text
DATA110-Capstone/
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   ├── README.md
│   └── Aman_Undirwade_DATA110_Capstone_Predictive_Maintenance.ipynb
├── src/
│   ├── predictive_maintenance.py
│   └── run_experiment.py
├── labs/
├── results/
├── figures/
├── assignments/
└── intermediate/
```

## Data
The raw NASA C-MAPSS FD004 benchmark is intentionally not redistributed in this repository. See `data/README.md` and the notebook for placement instructions.

## Running the capstone

```bash
pip install -r requirements.txt
python src/run_experiment.py --data-dir data/raw --dataset FD004
```

Or open the notebook under `notebooks/` and execute the cells sequentially.

## Academic-integrity / reproducibility note
The capstone implementation is intended to be understood and explained during the viva. The report describes a methodological, reproducible benchmark rather than a claim of a new ML algorithm. The FD004 benchmark is simulated and should not be interpreted as certified operational aircraft-maintenance evidence.

## GitHub checklist
Before submission, verify that:

- the repository is public and opens correctly;
- the capstone notebook is visible and readable;
- `src/` contains the reusable implementation;
- `requirements.txt` contains the required Python packages;
- raw NASA data are kept out of the repository;
- the report/PPTX/video are submitted through the required course channel if requested;
- no secrets, API keys, passwords or private credentials are committed.
