import pandas as pd


def normalize(input_df: pd.DataFrame) -> pd.DataFrame:
    output_df = input_df.copy()
    cols_exclude = ["subject_id", "trial", "status"]
    cols_norm = [col for col in input_df.columns if col not in cols_exclude]

    output_df[cols_norm] = input_df[cols_norm].apply(
        lambda x: (x - x.min()) / (x.max() - x.min()), axis=0
    )

    return output_df
