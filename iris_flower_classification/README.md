## Iris Flower Classification

This project classifies iris flowers into Setosa, Versicolor, and Virginica using classic machine learning models.

### Contents
- `train.py`: End-to-end training and evaluation script
- `notebooks/iris_eda_and_modeling.ipynb`: EDA and modeling notebook
- `artifacts/`: Saved models, metrics, and plots

### Quickstart
1. Create and activate a virtual environment (Windows PowerShell):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Run training:
   ```powershell
   python train.py --test-size 0.2 --random-state 42
   ```

Artifacts (metrics, confusion matrices, and the best model) will be saved under `artifacts/`.

### Models
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree

### Evaluation
We report accuracy, precision, recall, F1, and confusion matrices for each model.


