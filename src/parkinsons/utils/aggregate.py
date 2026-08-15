import pandas as pd


def aggregate(input_df: pd.DataFrame, group_by_col: str) -> pd.DataFrame:
    cols = input_df.columns.difference(["trial"])
    return input_df[cols].groupby(group_by_col).mean().reset_index()
