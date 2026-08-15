import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def model_generator(
    features: pd.DataFrame,
    target: pd.Series,
    size: float,
    seed: int,
    n_neighbors: int,
) -> tuple:
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=size, random_state=seed
    )

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(x_train, y_train)

    train_predictions = model.predict(x_train)
    train_metrics = {
        "accuracy": accuracy_score(y_train, train_predictions),
        "f1_score": f1_score(y_train, train_predictions, average="weighted"),
        "precision": precision_score(y_train, train_predictions, average="weighted"),
        "recall": recall_score(y_train, train_predictions, average="weighted"),
    }

    test_predictions = model.predict(x_test)
    test_metrics = {
        "accuracy": accuracy_score(y_test, test_predictions),
        "f1_score": f1_score(y_test, test_predictions, average="weighted"),
        "precision": precision_score(y_test, test_predictions, average="weighted"),
        "recall": recall_score(y_test, test_predictions, average="weighted"),
    }

    return model, train_metrics, test_metrics


def best_k(
    features: pd.DataFrame,
    target: pd.Series,
    size: float,
    seed: int,
    neighbors_range: np.ndarray,
) -> tuple:
    train_accuracies = []
    test_accuracies = []
    metrics_dict = {}
    intersections = []

    for k in neighbors_range:
        _, train_metrics, test_metrics = model_generator(
            features, target, size, seed, k
        )
        train_accuracies.append(train_metrics["accuracy"])
        test_accuracies.append(test_metrics["accuracy"])
        metrics_dict[k] = {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        }

    for i in range(1, len(neighbors_range)):
        if (
            train_accuracies[i - 1] > test_accuracies[i - 1]
            and train_accuracies[i] <= test_accuracies[i]
        ) or (
            train_accuracies[i - 1] < test_accuracies[i - 1]
            and train_accuracies[i] >= test_accuracies[i]
        ):
            slope_train = (train_accuracies[i] - train_accuracies[i - 1]) / (
                neighbors_range[i] - neighbors_range[i - 1]
            )
            slope_test = (test_accuracies[i] - test_accuracies[i - 1]) / (
                neighbors_range[i] - neighbors_range[i - 1]
            )
            intercept_x = neighbors_range[i - 1] + (
                test_accuracies[i - 1] - train_accuracies[i - 1]
            ) / (slope_train - slope_test)
            intercept_y = train_accuracies[i - 1] + slope_train * (
                intercept_x - neighbors_range[i - 1]
            )
            intersections.append((intercept_x, intercept_y))

    if intersections:
        best_intersection = max(intersections, key=lambda x: x[1])
        best_k_value = best_intersection[0]
    else:
        best_k_value = None

    return best_k_value, metrics_dict, intersections
