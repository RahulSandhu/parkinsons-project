import itertools
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def collinearity(df: pd.DataFrame, feature_groups: dict) -> dict:
    """
    Analyzes feature collinearity within defined groups, selects the least
    correlated features, and compares their correlations with others in the
    group.

    Inputs:
        - df (pd.DataFrame): The input DataFrame.
        - feature_groups (dict): A dictionary where keys are group names and
          values are lists of feature names.

    Outputs:
        - dict: A dictionary containing the results for each feature group.
    """
    # Preallocate results dictionary
    results = {}

    # Compute and order the correlation matrix for the group
    for group_name, features in feature_groups.items():
        correlation_matrix = df[features].corr().abs()  # type: ignore[arg-type]
        ordered_correlation = (
            correlation_matrix.where(~correlation_matrix.eq(1))
            .stack()
            .reset_index()
            .rename(
                columns={
                    0: "Correlation",
                    "level_0": "Feature 1",
                    "level_1": "Feature 2",
                }
            )
            .sort_values(by="Correlation", ascending=True)
        )
        lowest_correlation = ordered_correlation.iloc[0]

        # Extract feature 1 and 2
        feature1 = lowest_correlation["Feature 1"]
        feature2 = lowest_correlation["Feature 2"]
        scores = {feature: 0 for feature in features}

        # Compare feature 1 and feature 2
        for feature in features:
            if feature != feature1 and feature != feature2:
                if (
                    correlation_matrix.loc[feature1, feature]
                    < correlation_matrix.loc[feature2, feature]
                ):
                    scores[feature1] += 1
                    scores[feature2] -= 1
                else:
                    scores[feature2] += 1
                    scores[feature1] -= 1
        results[group_name] = (
            feature1 if scores[feature1] > scores[feature2] else feature2
        )

    # Concatenate the removing features
    remove_features = []
    for group_name, features in feature_groups.items():
        remove_features.extend([f for f in features if f != results[group_name]])

    return remove_features  # type: ignore[arg-type]


def scatter_plot(
    df: pd.DataFrame,
    group_name: str,
    features: list,
    output_dir: str,
) -> None:
    """
    Plots scatter plots for all combinations of features within a group.

    Inputs:
        - df (pd.DataFrame): The input DataFrame.
        - group_name (str): Name of the feature group.
        - features (list): List of feature names in the group.
        - output_dir (str): Directory to save the plot.

    Outputs:
        - fig: Scatter plot saved in the specified directory.
    """
    # Number of pairs for correlation
    combinations = list(itertools.combinations(features, 2))
    n_combinations = len(combinations)

    # Determine the grid size
    n_cols = min(3, n_combinations)
    n_rows = (n_combinations + n_cols - 1) // n_cols

    # Define axes
    _, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten() if n_combinations > 1 else [axes]

    # Scatter plot
    for i, (feature1, feature2) in enumerate(combinations):
        ax = axes[i]
        sns.scatterplot(data=df, x=feature1, y=feature2, hue="status", alpha=0.7, ax=ax)
        ax.set_title(f"{feature1} vs {feature2}")
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.legend(title="Status")

    # Turn off unused axes
    for ax in axes[len(combinations) :]:
        ax.axis("off")

    # Adjust layout and save figure
    plt.tight_layout()
    output_file = os.path.join(
        output_dir, f"{group_name.replace(' ', '_')}.png".lower()
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Use custom style
    plt.style.use("../../config/custom.mplstyle")

    # Load dataset
    df_norm = pd.read_csv("../../data/processed/parkinsons_norm.data")

    # Define output directory and ensure it exists
    output_dir = "../../images/feature_selection/"
    os.makedirs(output_dir, exist_ok=True)

    # Define feature groups
    feature_groups = {
        "Fundamental Frequency": ["avFF", "maxFF", "minFF"],
        "Jitter": ["absJitter", "percJitter", "rap", "ppq", "ddp"],
        "Shimmer": ["lShimmer", "dbShimmer", "apq3", "apq5", "apq", "dda"],
    }

    # Generate scatter plots for each group
    for group_name, features in feature_groups.items():
        scatter_plot(df_norm, group_name, features, output_dir)

    # Assess collinearity and get features to remove
    remove_features = collinearity(df_norm, feature_groups)
    final_features = set(df_norm.columns) - set(remove_features)
    final_features = list(final_features)
    final_features.remove("subject_id")
    final_features.remove("trial")
    final_features.remove("status")

    print("Final features:", final_features)
