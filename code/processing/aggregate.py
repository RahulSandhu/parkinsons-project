import os

import pandas as pd
from processing.outliers import outliers
from processing.rename import rename


def aggregate(input_df, group_by_col) -> pd.DataFrame:
    """
    Aggregates the DataFrame by calculating the mean of all columns grouping
    the column by the specified feature.

    Inputs:
        - input_df (pd.DataFrame): The input DataFrame.
        - group_by_col (str): The column to group by.

    Outputs:
        - pd.DataFrame: The aggregated DataFrame with mean values.
    """
    # Select all columns except trial
    cols = input_df.columns.difference(["trial"])

    # Group by the specified column and calculate the mean
    output_df = input_df[cols].groupby(group_by_col).mean().reset_index()

    return output_df


if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("../../data/raw/parkinsons.data")

    # Rename columns
    df_rename = rename(df)

    # Remove outliers
    df_clean = outliers(df_rename)

    # Aggregate by subject_id
    df_avg = aggregate(df_clean, "subject_id")

    # Define output directory and file path
    output_dir = "../../data/processed"
    output_file = os.path.join(output_dir, "parkinsons_avg.data")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the cleaned DataFrame to the specified file
    df_avg.to_csv(output_file, index=False)
