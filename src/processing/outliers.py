import os

import pandas as pd
from processing.rename import rename


def outliers(input_df) -> pd.DataFrame:
    """
    Removes outliers within each subject group using the IQR method. Outliers
    are replaced with the mean of the non-outlier values for that subject.

    Inputs:
        - input_df (pd.DataFrame): The input DataFrame with 'subject_id' and
          numeric trial data.

    Outputs:
        - pd.DataFrame: A DataFrame with outliers replaced within each subject
          group.
    """
    # Create a copy of the input DataFrame
    output_df = input_df.copy()

    # Process each subject group
    for _, group in output_df.groupby("subject_id"):
        numeric_cols = group.select_dtypes(include="number").columns

        # Calculate Q1, Q3, and IQR for the column within each subject
        for column in numeric_cols:
            Q1 = group[column].quantile(0.25)
            Q3 = group[column].quantile(0.75)
            IQR = Q3 - Q1

            # Determine outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Replace outliers with the mean of non-outlier values for this subject
            non_outlier_mean = group.loc[
                (group[column] >= lower_bound) & (group[column] <= upper_bound), column
            ].mean()
            outlier_indices = group.index[
                (group[column] < lower_bound) | (group[column] > upper_bound)
            ]

            # Replace outliers
            output_df.loc[outlier_indices, column] = non_outlier_mean

    return output_df


if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("../../data/raw/parkinsons.data")

    # Rename columns
    df_rename = rename(df)

    # Remove outliers
    df_clean = outliers(df_rename)

    # Define output directory and file path
    output_dir = "../../data/processed"
    output_file = os.path.join(output_dir, "parkinsons_clean.data")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the cleaned DataFrame to the specified file
    df_clean.to_csv(output_file, index=False)
