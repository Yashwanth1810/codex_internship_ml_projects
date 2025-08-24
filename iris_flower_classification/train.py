import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def load_iris_dataframe() -> pd.DataFrame:
    iris = datasets.load_iris(as_frame=True)
    df = iris.frame.copy()
    df.rename(columns={"target": "species"}, inplace=True)
    df["species_name"] = df["species"].map(dict(enumerate(iris.target_names)))
    return df


def create_eda_plots(df: pd.DataFrame, artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Pairplot colored by species
    pairplot_path = artifacts_dir / "pairplot.png"
    sns.pairplot(
        df,
        vars=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
        hue="species_name",
        corner=True,
        diag_kind="hist",
    )
    plt.tight_layout()
    plt.savefig(pairplot_path, dpi=150)
    plt.close()

    # Histograms per feature
    hist_dir = artifacts_dir / "histograms"
    hist_dir.mkdir(exist_ok=True)
    feature_columns = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]
    for feature in feature_columns:
        plt.figure(figsize=(5, 3))
        sns.histplot(df, x=feature, hue="species_name", kde=True, element="step")
        plt.title(f"Histogram: {feature}")
        plt.tight_layout()
        plt.savefig(hist_dir / f"{feature.replace(' ', '_').replace('(', '').replace(')', '')}.png", dpi=150)
        plt.close()


def split_data(df: pd.DataFrame, test_size: float, random_state: int):
    feature_columns = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]
    X = df[feature_columns]
    y = df["species"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def build_models():
    numeric_features = [0, 1, 2, 3]
    scaler = StandardScaler()

    preprocessing = ColumnTransformer(
        transformers=[
            ("scale", scaler, numeric_features),
        ],
        remainder="drop",
    )

    models = {
        "log_reg": Pipeline(
            steps=[
                ("preprocess", preprocessing),
                ("clf", LogisticRegression(max_iter=1000, multi_class="auto")),
            ]
        ),
        "knn": Pipeline(
            steps=[
                ("preprocess", preprocessing),
                ("clf", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "decision_tree": Pipeline(
            steps=[
                ("preprocess", preprocessing),
                ("clf", DecisionTreeClassifier(random_state=0)),
            ]
        ),
    }
    return models


def evaluate_and_save(name: str, model: Pipeline, X_test, y_test, artifacts_dir: Path) -> dict:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_text = classification_report(y_test, y_pred)

    metrics_path = artifacts_dir / f"metrics_{name}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "report": report}, f, indent=2)

    # Save human-readable report
    with open(artifacts_dir / f"report_{name}.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(artifacts_dir / f"confusion_matrix_{name}.png", dpi=150)
    plt.close()

    return {"name": name, "accuracy": acc}


def save_best_model(best_name: str, best_model: Pipeline, artifacts_dir: Path) -> None:
    output_path = artifacts_dir / f"best_model_{best_name}.joblib"
    joblib.dump(best_model, output_path)


def main():
    parser = argparse.ArgumentParser(description="Train Iris classifiers and save artifacts")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-eda", action="store_true", help="Skip EDA plots")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Load and EDA
    df = load_iris_dataframe()
    if not args.no_eda:
        create_eda_plots(df, artifacts_dir)

    # Split
    X_train, X_test, y_train, y_test = split_data(df, test_size=args.test_size, random_state=args.random_state)

    # Build models
    models = build_models()

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        result = evaluate_and_save(name, model, X_test, y_test, artifacts_dir)
        results.append((result["accuracy"], name, model))

    # Determine best
    results.sort(reverse=True, key=lambda x: x[0])
    best_acc, best_name, best_model = results[0]

    # Save best model and summary
    save_best_model(best_name, best_model, artifacts_dir)
    with open(artifacts_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, "best_accuracy": best_acc}, f, indent=2)

    print(f"Best model: {best_name} with accuracy={best_acc:.4f}")


if __name__ == "__main__":
    main()


