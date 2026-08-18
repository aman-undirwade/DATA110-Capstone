# DATA110 Capstone — Aircraft Engine Predictive Maintenance

## Student
**Aman Undirwade**

## Course Portfolio
This repository is organized as the DATA110 machine-learning portfolio/capstone repository. It contains the existing aircraft-engine predictive-maintenance project plus a structured set of course lab/practice implementations covering the major topics repeatedly taught and practiced in class.

## Existing Capstone Project
The capstone studies early aircraft-engine failure detection using NASA C-MAPSS FD004. It compares Logistic Regression, Gaussian Naive Bayes, KNN, Decision Tree and Random Forest, uses engine-aware validation, evaluates warning horizons, and documents the selected Random Forest operating point.

## Course/Lab Coverage
The `labs/` folder now contains compact, runnable practice implementations for:

- Python + end-to-end ML workflow
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machine (SVM)
- K-Means clustering
- Hierarchical/Agglomerative clustering
- PCA
- ANN fundamentals and weight update/backpropagation concepts
- NLP text preprocessing
- Exam numerical practice for K-Means, KNN, Decision Tree, PCA, ANN and SVM

## Repository Structure

```text
DATA110-Capstone/
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   └── README.md
├── src/
│   ├── predictive_maintenance.py
│   └── run_experiment.py
├── labs/
│   ├── README.md
│   ├── 01_python_ml_template.py
│   ├── 02_classification_algorithms.py
│   ├── 03_clustering_and_pca.py
│   ├── 04_ann_and_backpropagation.py
│   ├── 05_nlp_text_preprocessing.py
│   └── 06_exam_numerical_practice.md
├── results/
├── figures/
├── assignments/
└── intermediate/
```

## Important academic-integrity note
The lab files in this repository are practice implementations/templates built from the course topics. They do **not** claim to be original historical submissions unless an actual submitted lab was already present in the repository. Dataset-specific labs should be run with the dataset supplied by the course before submission.

## Running the Capstone

```bash
pip install -r requirements.txt
python src/run_experiment.py --data-dir data/raw --dataset FD004
```

The raw NASA benchmark is not redistributed in this repository. See `data/README.md` for placement instructions.

## GitHub Exam Checklist
Before the exam, verify that:

- the repository is public and opens correctly;
- the capstone project is visible;
- the `labs/` folder is visible;
- Python/ML, classification, clustering/PCA, ANN, NLP and numerical-practice files are present;
- actual course-submitted lab files are added if they were submitted separately;
- no secrets, API keys, passwords or private credentials are committed.
