import os
from typing import cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from model.collinearity import collinearity


def model_generator(
    X: pd.DataFrame,
    Y: pd.Series,
    size: float,
    seed: int,
    n_neighbors: int,
) -> tuple:
    """
    Train a KNN model with a user-defined number of neighbors.

    Inputs:
        - X (pd.DataFrame): Feature matrix.
        - Y (pd.Series): Target variable.
        - size (float): Test set size as a proportion.
        - seed (int): Random seed for reproducibility.
        - n_neighbors (int): Number of neighbors for KNN.

    Outputs:
        - model: The trained KNN model.
        - train_metrics (dict): Metrics on the training set.
        - test_metrics (dict): Metrics on the test set.
    """
    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=size, random_state=seed
    )

    # Train the model
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, Y_train)

    # Evaluate the model on training data
    train_predictions = model.predict(X_train)
    train_metrics = {
        "accuracy": accuracy_score(Y_train, train_predictions),
        "f1_score": f1_score(Y_train, train_predictions, average="weighted"),
        "precision": precision_score(Y_train, train_predictions, average="weighted"),
        "recall": recall_score(Y_train, train_predictions, average="weighted"),
    }

    # Evaluate the model on testing data
    test_predictions = model.predict(X_test)
    test_metrics = {
        "accuracy": accuracy_score(Y_test, test_predictions),
        "f1_score": f1_score(Y_test, test_predictions, average="weighted"),
        "precision": precision_score(Y_test, test_predictions, average="weighted"),
        "recall": recall_score(Y_test, test_predictions, average="weighted"),
    }

    return model, train_metrics, test_metrics


def best_k(
    X: pd.DataFrame,
    Y: pd.Series,
    size: float,
    seed: int,
    neighbors_range: np.ndarray,
) -> tuple:
    """
    Find the best number of neighbors (k) based on testing accuracy and
    calculate intersections.

    Inputs:
        - X (DataFrame): Feature matrix.
        - Y (Series): Target variable.
        - size (float): Test set size as a proportion.
        - seed (int): Random seed for reproducibility.
        - neighbors_range (array): Range of neighbor values to evaluate.

    Outputs:
        - best_k_value (float): The k value with the highest testing accuracy.
        - metrics_dict (dict): Training and testing metrics for each k.
        - intersections (list): Intersection points as (k, accuracy).
    """
    # Preallocate arrays
    train_accuracies = []
    test_accuracies = []
    metrics_dict = {}
    intersections = []

    # Generate different models for different k
    for k in neighbors_range:
        _, train_metrics, test_metrics = model_generator(X, Y, size, seed, k)
        train_accuracies.append(train_metrics["accuracy"])
        test_accuracies.append(test_metrics["accuracy"])
        metrics_dict[k] = {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        }

    # Find intersections
    for i in range(1, len(neighbors_range)):
        if (
            train_accuracies[i - 1] > test_accuracies[i - 1]
            and train_accuracies[i] <= test_accuracies[i]
        ) or (
            train_accuracies[i - 1] < test_accuracies[i - 1]
            and train_accuracies[i] >= test_accuracies[i]
        ):
            # Linear interpolation to find the approximate intersection
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

    # Select the best k value
    if intersections:
        best_intersection = max(intersections, key=lambda x: x[1])
        best_k = best_intersection[0]
    else:
        best_k = None

    return best_k, metrics_dict, intersections


if __name__ == "__main__":
    # Fix for Wayland
    matplotlib.use("QtAgg")

    # Use custom style
    plt.style.use("../../config/matplotlib/mhedas.mplstyle")

    # Load preprocessed data
    df_clean = pd.read_csv("../../data/processed/parkinsons_clean.data")
    df_avg = pd.read_csv("../../data/processed/parkinsons_avg.data")
    df_norm = pd.read_csv("../../data/processed/parkinsons_norm.data")

    # Define feature groups
    feature_groups = {
        "Fundamental Frequency": ["avFF", "maxFF", "minFF"],
        "Jitter": ["absJitter", "percJitter", "rap", "ppq", "ddp"],
        "Shimmer": ["lShimmer", "dbShimmer", "apq3", "apq5", "apq", "dda"],
    }

    # Final features
    remove_features = collinearity(df_norm, feature_groups)
    final_features = set(df_norm.columns) - set(remove_features)
    final_features = list(final_features)
    final_features.remove("subject_id")
    final_features.remove("trial")
    final_features.remove("status")

    # Define neighbors range
    neighbors_range = np.arange(1, 22)

    # Analyze for df_clean
    X_clean = df_clean[final_features]
    Y_clean = df_clean["status"]
    X_clean = cast(pd.DataFrame, X_clean)
    Y_clean = cast(pd.Series, Y_clean)
    best_k_clean, metrics_clean, intersections_clean = best_k(
        X_clean,
        Y_clean,
        size=0.3,
        seed=123,
        neighbors_range=neighbors_range,
    )
    print(f"Best k for df_clean: {best_k_clean}")
    print(f"Intersections for df_clean: {intersections_clean}")

    # Analyze for df_avg
    X_avg = df_avg[final_features]
    Y_avg = df_avg["status"]
    X_avg = cast(pd.DataFrame, X_avg)
    Y_avg = cast(pd.Series, Y_avg)
    best_k_avg, metrics_avg, intersections_avg = best_k(
        X_avg,
        Y_avg,
        size=0.3,
        seed=123,
        neighbors_range=neighbors_range,
    )
    print(f"Best k for df_avg: {best_k_avg}")
    print(f"Intersections for df_avg: {intersections_avg}")

    # Analyze for df_norm
    X_norm = df_norm[final_features]
    Y_norm = df_norm["status"]
    X_norm = cast(pd.DataFrame, X_norm)
    Y_norm = cast(pd.Series, Y_norm)
    best_k_norm, metrics_norm, intersections_norm = best_k(
        X_norm,
        Y_norm,
        size=0.3,
        seed=123,
        neighbors_range=neighbors_range,
    )
    print(f"Best k for df_norm: {best_k_norm}")
    print(f"Intersections for df_norm: {intersections_norm}")

    # Define output directory and ensure it exists
    output_dir = "../../images/feature_selection/"
    os.makedirs(output_dir, exist_ok=True)

    # Plot results for all datasets
    datasets = [
        {
            "name": "df_clean",
            "metrics": metrics_clean,
            "best_k": best_k_clean,
            "intersections": intersections_clean,
        },
        {
            "name": "df_avg",
            "metrics": metrics_avg,
            "best_k": best_k_avg,
            "intersections": intersections_avg,
        },
        {
            "name": "df_norm",
            "metrics": metrics_norm,
            "best_k": best_k_norm,
            "intersections": intersections_norm,
        },
    ]

    # Create figure
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot best neighbors value for each dataset
    for i, dataset in enumerate(datasets):
        axs[i].plot(
            neighbors_range,
            [
                dataset["metrics"][k]["train_metrics"]["accuracy"]
                for k in neighbors_range
            ],
            label="Training Accuracy",
        )
        axs[i].plot(
            neighbors_range,
            [
                dataset["metrics"][k]["test_metrics"]["accuracy"]
                for k in neighbors_range
            ],
            label="Testing Accuracy",
        )

        # Add intersection line for best_k
        if dataset["intersections"]:
            axs[i].axvline(
                x=dataset["best_k"],
                linestyle="--",
                color="red",
            )
            for x, y in dataset["intersections"]:
                axs[i].scatter(x, y, color="red")
                if x == dataset["best_k"]:
                    axs[i].axhline(
                        y=y,
                        linestyle="--",
                        color="red",
                        label=f"Best k = {x:.2f}, Accuracy = {y:.2f}",
                    )
        axs[i].set_title(f"{dataset['name']}")
        axs[i].set_xlabel("Number of Neighbors")
        axs[i].set_ylabel("Accuracy")
        axs[i].legend()

    # Adjust layout and save figure
    plt.tight_layout()
    output_file = os.path.join(output_dir, "best_k.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
