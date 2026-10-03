# Iris Flower Species Classification Using Machine Learning

## Objective

This beginner project uses four measurements of an Iris flower to predict its species: Setosa, Versicolor, or Virginica.

## Introduction

Classification is a type of machine learning where a computer learns from examples that already have categories, then assigns a category to a new example. Here, the categories are the three Iris species.

The Iris dataset is useful for learning because it is small, well-known, has four understandable measurements, and contains examples from all three species. It is included with scikit-learn, so there is no separate data download.

## Dataset description

The project uses scikit-learn's built-in dataset, with 150 flower samples, four numeric measurement features, and three species classes. Measurements are in centimeters:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The script displays the first rows, dataset dimensions and class names, column information, and missing-value counts. The built-in dataset has no missing measurements.

The `Iris.csv` file supplied alongside the request is a separate copy of Iris data. The implementation deliberately loads the built-in scikit-learn version because that is the dataset source specified for this assignment.

## Algorithm: K-Nearest Neighbors

K-Nearest Neighbors (KNN) compares a new flower's measurements with flowers in the training set. It finds the five closest examples and predicts the species that appears most often among them. This makes KNN a straightforward first classification algorithm to understand.

## Data preprocessing and training

The script separates the four measurements (`X`) from the species labels (`y`). It splits the data into 80% training data and 20% testing data. The training portion teaches the model; the test portion is held back and used to estimate how it performs on unseen examples. A fixed random seed makes the split repeatable, and stratification preserves the class proportions in both sets.

KNN is trained with `fit()` on the training measurements and labels. The script then predicts labels for the test measurements.

## Evaluation

The program prints accuracy, macro precision, macro recall, macro F1-score, a confusion matrix, and a classification report. Macro averages calculate each species' score and then average the three scores equally.

- **Accuracy** is the proportion of test flowers whose predicted species matches the known species.
- **Precision** tells us, for a species, how often predictions of that species are correct.
- **Recall** tells us how many of the actual flowers of a species the model found.
- **F1-score** balances precision and recall.
- **Confusion matrix** compares actual labels (rows) with predicted labels (columns). Values on the diagonal are correct predictions; off-diagonal values show which species were mixed up.

The exact results are printed when the script runs. With the fixed 80:20 stratified split and these settings, the score is repeatable in a given scikit-learn environment. Strong scores mean the model separated the species well on this held-out sample, though a single split is not a guarantee of performance on every future flower.

## Visualizations

The script creates three figures with labeled axes and legends:

1. `iris_feature_relationships.png` — pairwise relationships between all four measurements, colored by species.
2. `iris_scatter_plot.png` — petal length versus petal width by species.
3. `iris_confusion_matrix.png` — a heatmap of actual and predicted species counts.

## Sample predictions

Three example flowers are classified in the script. It then offers an interactive prompt where you can enter sepal length, sepal width, petal length, and petal width in centimeters. Press Enter at the first prompt to skip interactive input.

## Run the project

Install the required libraries:

```bash
python -m pip install pandas numpy scikit-learn matplotlib seaborn
```

Run the script from this folder:

```bash
python iris_classification.py
```

The script prints the dataset overview and evaluation results, saves the three figures in the current folder, opens each figure, prints sample predictions, and offers an interactive prediction.

## Conclusion

This project demonstrates the basic classification workflow: load labeled examples, inspect the data, split it into training and test sets, fit a model, and measure its predictions. KNN uses nearby examples to make a species prediction. Accuracy summarizes the share of correct test predictions, while precision, recall, F1-score, and the confusion matrix explain performance for each species in more detail. The plots show that petal measurements are especially useful for distinguishing the three Iris species.
