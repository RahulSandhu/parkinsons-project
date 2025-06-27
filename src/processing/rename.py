import pandas as pd

# Define the column renaming dictionary
DICT_NAMES = {
    "MDVP:Fo(Hz)": "avFF",
    "MDVP:Fhi(Hz)": "maxFF",
    "MDVP:Flo(Hz)": "minFF",
    "MDVP:Jitter(%)": "percJitter",
    "MDVP:Jitter(Abs)": "absJitter",
    "MDVP:RAP": "rap",
    "MDVP:PPQ": "ppq",
    "Jitter:DDP": "ddp",
    "MDVP:Shimmer": "lShimmer",
    "MDVP:Shimmer(dB)": "dbShimmer",
    "Shimmer:APQ3": "apq3",
    "Shimmer:APQ5": "apq5",
    "MDVP:APQ": "apq",
    "Shimmer:DDA": "dda",
}


def rename(input_df, rename_dict=DICT_NAMES) -> pd.DataFrame:
    """
    Renames columns and rearranges the columns in the DataFrame.

    Inputs:
        - input_df (pd.DataFrame): The input DataFrame.
        - rename_dict (dict): A dictionary mapping old column names to new
          ones.

    Outputs:
        - pd.DataFrame: The renamed DataFrame.
    """
    # Create a copy of the input DataFrame
    output_df = input_df.copy()

    # Rename columns
    output_df = output_df.rename(columns=rename_dict)

    # Extract subject_id and trial from the name column
    for i, row in output_df.iterrows():
        split_name = row["name"].split("_")
        output_df.at[i, "subject_id"] = split_name[2]
        output_df.at[i, "trial"] = split_name[3]

    # Drop the name column
    output_df.drop(columns=["name"], inplace=True)

    # Rearrange columns
    cols_order = ["subject_id", "trial"] + [
        col for col in output_df.columns if col not in ["subject_id", "trial"]
    ]
    output_df = output_df[cols_order]

    return output_df


if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("../../data/raw/parkinsons.data")

    # Exploratory analysis
    print(df.head())
    print(df.info())
    print(df.describe())

    # Rename columns
    df_rename = rename(df)

    # Print column names
    print(df_rename.columns.tolist())

    # Count total number of observations
    total_observations = df_rename.shape[0]

    print(f"Total number of observations: {total_observations}")

    # Group by subject_id to get unique patients
    grouped_df = df_rename.groupby("subject_id").agg({"status": "first"}).reset_index()

    # Count total number of patients and controls
    patient_count = grouped_df[grouped_df["status"] == 1].shape[0]
    control_count = grouped_df[grouped_df["status"] == 0].shape[0]

    print(f"Total number of unique patients: {patient_count}")
    print(f"Total number of unique controls: {control_count}")

    # Count trials per subject_id
    trials_per_subject = df_rename.groupby("subject_id").size()

    print("Number of trials per subject:")
    print(trials_per_subject)
