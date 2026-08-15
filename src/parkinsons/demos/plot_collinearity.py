import itertools

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from parkinsons.utils.config import FIGURES_DIR


def scatter_plot(df: pd.DataFrame, group_name: str, features: list) -> None:
    combinations = list(itertools.combinations(features, 2))
    n_combinations = len(combinations)

    n_cols = min(3, n_combinations)
    n_rows = (n_combinations + n_cols - 1) // n_cols

    _, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten() if n_combinations > 1 else [axes]

    for i, (feature1, feature2) in enumerate(combinations):
        ax = axes[i]
        sns.scatterplot(data=df, x=feature1, y=feature2, hue="status", alpha=0.7, ax=ax)
        ax.set_title(f"{feature1} vs {feature2}")
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.legend(title="Status")

    for ax in axes[len(combinations) :]:
        ax.axis("off")

    plt.tight_layout()
    output_file = FIGURES_DIR / f"{group_name.replace(' ', '_')}.png".lower()
    plt.savefig(output_file)
    plt.close()


def main():
    df_norm = pd.read_csv("data/processed/parkinsons_norm.data")

    feature_groups = {
        "Fundamental Frequency": ["avFF", "maxFF", "minFF"],
        "Jitter": ["absJitter", "percJitter", "rap", "ppq", "ddp"],
        "Shimmer": ["lShimmer", "dbShimmer", "apq3", "apq5", "apq", "dda"],
    }

    for group_name, features in feature_groups.items():
        scatter_plot(df_norm, group_name, features)


if __name__ == "__main__":
    main()
