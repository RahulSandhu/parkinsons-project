import pandas as pd


def collinearity(df: pd.DataFrame, feature_groups: dict) -> list:
    results = {}

    for group_name, features in feature_groups.items():
        correlation_matrix = df[features].corr().abs()
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

        feature1 = lowest_correlation["Feature 1"]
        feature2 = lowest_correlation["Feature 2"]
        scores = {feature: 0 for feature in features}

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

    remove_features = []
    for group_name, features in feature_groups.items():
        remove_features.extend([f for f in features if f != results[group_name]])

    return remove_features
