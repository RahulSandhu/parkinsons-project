import os

import pandas as pd
from processing.outliers import outliers
from processing.rename import rename


def normalize(input_df) -> pd.DataFrame:
    """
    Normalizes numeric columns in the DataFrame by scaling values to the range
    [0, 1]. Excludes specified columns from normalization.

    Inputs:
        - input_df (pd.DataFrame): The input DataFrame.

    Outputs:
        - pd.DataFrame: A DataFrame with normalized values for selected
          columns.
    """
    # Create a copy of the original DataFrame
    output_df = input_df.copy()

    # Columns to exclude from normalization
    cols_exclude = ["subject_id", "trial", "status"]

    # Identify columns to normalize
    cols_norm = [col for col in input_df.columns if col not in cols_exclude]

    # Normalize only the selected columns
    output_df[cols_norm] = input_df[cols_norm].apply(
        lambda x: (x - x.min()) / (x.max() - x.min()), axis=0
    )

    return output_df


if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("../../data/raw/parkinsons.data")

    # Rename columns
    df_rename = rename(df)

    # Remove outliers
    df_clean = outliers(df_rename)

    # Normalize DataFrame
    df_norm = normalize(df_clean)

    # Define output directory and file path
    output_dir = "../../data/processed"
    output_file = os.path.join(output_dir, "parkinsons_norm.data")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the cleaned DataFrame to the specified file
    df_norm.to_csv(output_file, index=False)
