"""Iris Flower Species Classification Using Machine Learning.

Run with: python iris_classification.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main():
    # Load scikit-learn's built-in Iris dataset and give its columns readable names.
    iris = load_iris()
    feature_names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
    species_names = [f"Iris {name.title()}" for name in iris.target_names]
    data = pd.DataFrame(iris.data, columns=feature_names)
    data["Species"] = [species_names[index] for index in iris.target]

    print("IRIS FLOWER SPECIES CLASSIFICATION")
    print("\nFirst five rows:")
    print(data.head())
    print(f"\nSamples: {len(data)}")
    print(f"Features: {len(feature_names)} ({', '.join(feature_names)})")
    print(f"Classes: {data['Species'].nunique()} ({', '.join(species_names)})")
    print("\nDataset information:")
    data.info()
    print("\nMissing values in each column:")
    print(data.isnull().sum())

    # X contains measurements; y contains the known species labels.
    X = data[feature_names]
    y = data["Species"]

    # Keep 20% of the data aside for evaluation; stratify keeps class proportions even.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # KNN predicts the label most common among the k closest training examples.
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    # Evaluate predictions against the correct labels held out from training.
    y_pred = model.predict(X_test)
    labels = species_names
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, labels=labels, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, labels=labels, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0)

    print("\nModel: K-Nearest Neighbors (k=5)")
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Macro precision: {precision:.2%}")
    print(f"Macro recall: {recall:.2%}")
    print(f"Macro F1-score: {f1:.2%}")
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(matrix, index=labels, columns=labels))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=labels, zero_division=0))

    # A pair plot shows how pairs of measurements vary across species.
    sns.pairplot(data, hue="Species", corner=True, diag_kind="hist")
    plt.suptitle("Iris Measurements by Species", y=1.02)
    plt.savefig("iris_feature_relationships.png", dpi=150, bbox_inches="tight")
    plt.show()

    # A labeled scatter plot gives one easy-to-read feature comparison.
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=data, x="Petal Length", y="Petal Width", hue="Species", s=65
    )
    plt.title("Petal Length and Width by Iris Species")
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.legend(title="Species")
    plt.tight_layout()
    plt.savefig("iris_scatter_plot.png", dpi=150)
    plt.show()

    # Heatmap makes correct and incorrect classifications easy to spot.
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels
    )
    plt.title("Confusion Matrix for the Test Data")
    plt.xlabel("Predicted species")
    plt.ylabel("Actual species")
    plt.tight_layout()
    plt.savefig("iris_confusion_matrix.png", dpi=150)
    plt.show()

    # Example predictions. Measurements are in centimeters, in the feature order above.
    examples = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2], [6.0, 2.9, 4.5, 1.5], [6.5, 3.0, 5.5, 2.0]],
        columns=feature_names,
    )
    print("\nSample predictions:")
    for measurements, prediction in zip(examples.to_numpy(), model.predict(examples)):
        print(f"{measurements.tolist()} -> {prediction}")

    # Optional interactive prediction. Press Enter at the first prompt to skip.
    print("\nTry a prediction with your own measurements (in cm).")
    first_value = input("Sepal Length (or press Enter to finish): ").strip()
    if first_value:
        try:
            values = [float(first_value)]
            for feature in feature_names[1:]:
                values.append(float(input(f"{feature} (cm): ")))
            if not np.isfinite(values).all():
                raise ValueError("Measurements must be finite numbers.")
            new_flower = pd.DataFrame([values], columns=feature_names)
            prediction = model.predict(new_flower)[0]
            print(f"Predicted species: {prediction}")
        except ValueError as error:
            print(f"Please enter four valid numeric measurements. ({error})")
    else:
        print("Interactive prediction skipped.")

    print("\nThe test accuracy is the share of held-out flowers classified correctly.")
    print("A high score means the model performed well on this test split; results can vary with another split.")


if __name__ == "__main__":
    main()
