import pandas as pd


def outliers(input_df: pd.DataFrame) -> pd.DataFrame:
    output_df = input_df.copy()

    for _, group in output_df.groupby("subject_id"):
        numeric_cols = group.select_dtypes(include="number").columns

        for column in numeric_cols:
            q1 = group[column].quantile(0.25)
            q3 = group[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            non_outlier_mean = group.loc[
                (group[column] >= lower_bound) & (group[column] <= upper_bound), column
            ].mean()
            outlier_indices = group.index[
                (group[column] < lower_bound) | (group[column] > upper_bound)
            ]
            output_df.loc[outlier_indices, column] = non_outlier_mean

    return output_df
