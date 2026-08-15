import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from parkinsons.analysis.model import best_k
from parkinsons.utils.config import FIGURES_DIR


def plot_dataset(ax, dataset, neighbors_range):
    metrics = dataset["metrics"]

    ax.plot(
        neighbors_range,
        [metrics[k]["train_metrics"]["accuracy"] for k in neighbors_range],
        label="Training Accuracy",
    )
    ax.plot(
        neighbors_range,
        [metrics[k]["test_metrics"]["accuracy"] for k in neighbors_range],
        label="Testing Accuracy",
    )

    if dataset["intersections"]:
        ax.axvline(x=dataset["best_k"], linestyle="--", color="red")
        for x, y in dataset["intersections"]:
            ax.scatter(x, y, color="red")
            if x == dataset["best_k"]:
                ax.axhline(
                    y=y,
                    linestyle="--",
                    color="red",
                    label=f"Best k = {x:.2f}, Accuracy = {y:.2f}",
                )

    ax.set_title(dataset["name"])
    ax.set_xlabel("Number of Neighbors")
    ax.set_ylabel("Accuracy")
    ax.legend()


def main():
    df_clean = pd.read_csv("data/processed/parkinsons_clean.data")
    df_avg = pd.read_csv("data/processed/parkinsons_avg.data")
    df_norm = pd.read_csv("data/processed/parkinsons_norm.data")

    feature_groups = {
        "Fundamental Frequency": ["avFF", "maxFF", "minFF"],
        "Jitter": ["absJitter", "percJitter", "rap", "ppq", "ddp"],
        "Shimmer": ["lShimmer", "dbShimmer", "apq3", "apq5", "apq", "dda"],
    }

    # placeholder for final_features selection, should match main.py logic
    from parkinsons.utils.collinearity import collinearity

    remove_features = collinearity(df_norm, feature_groups)
    final_features = list(set(df_norm.columns) - set(remove_features))
    final_features.remove("subject_id")
    final_features.remove("trial")
    final_features.remove("status")

    neighbors_range = np.arange(1, 22)
    datasets = [
        {
            "name": "df_clean",
            "X": df_clean[final_features],
            "Y": df_clean["status"],
        },
        {
            "name": "df_avg",
            "X": df_avg[final_features],
            "Y": df_avg["status"],
        },
        {
            "name": "df_norm",
            "X": df_norm[final_features],
            "Y": df_norm["status"],
        },
    ]

    for dataset in datasets:
        best_k_value, metrics, intersections = best_k(
            dataset["X"],
            dataset["Y"],
            size=0.3,
            seed=123,
            neighbors_range=neighbors_range,
        )
        dataset["best_k"] = best_k_value
        dataset["metrics"] = metrics
        dataset["intersections"] = intersections
        print(f"Best k for {dataset['name']}: {best_k_value}")

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for i, dataset in enumerate(datasets):
        plot_dataset(axs[i], dataset, neighbors_range)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "best_k.png")
    plt.close()


if __name__ == "__main__":
    main()
