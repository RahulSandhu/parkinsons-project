import pandas as pd

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


def rename(input_df: pd.DataFrame, rename_dict: dict = DICT_NAMES) -> pd.DataFrame:
    output_df = input_df.copy().rename(columns=rename_dict)

    for i, row in output_df.iterrows():
        split_name = row["name"].split("_")
        output_df.at[i, "subject_id"] = split_name[2]
        output_df.at[i, "trial"] = split_name[3]

    output_df.drop(columns=["name"], inplace=True)

    cols_order = ["subject_id", "trial"] + [
        col for col in output_df.columns if col not in ["subject_id", "trial"]
    ]

    return output_df[cols_order]
