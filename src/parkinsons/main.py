import pickle
from datetime import datetime
from typing import cast

import pandas as pd

from parkinsons.analysis.model import model_generator
from parkinsons.utils.aggregate import aggregate
from parkinsons.utils.collinearity import collinearity
from parkinsons.utils.config import MODELS_DIR, PROCESSED_DIR, RAW_DIR
from parkinsons.utils.normalize import normalize
from parkinsons.utils.outliers import outliers
from parkinsons.utils.rename import rename

feature_groups = {
    "Fundamental Frequency": ["avFF", "maxFF", "minFF"],
    "Jitter": ["absJitter", "percJitter", "rap", "ppq", "ddp"],
    "Shimmer": ["lShimmer", "dbShimmer", "apq3", "apq5", "apq", "dda"],
}

n_neighbors = 5


def load_and_process() -> tuple:
    df = pd.read_csv(RAW_DIR / "parkinsons.data")
    df_rename = rename(df)
    df_clean = outliers(df_rename)
    df_avg = aggregate(df_clean, "subject_id")
    df_norm = normalize(df_clean)
    return df_clean, df_avg, df_norm


def select_features(df_norm: pd.DataFrame) -> list:
    remove_features = collinearity(df_norm, feature_groups)
    final_features = list(set(df_norm.columns) - set(remove_features))
    final_features.remove("subject_id")
    final_features.remove("trial")
    final_features.remove("status")
    return final_features


def save_processed_datasets(
    df_clean: pd.DataFrame, df_avg: pd.DataFrame, df_norm: pd.DataFrame
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(PROCESSED_DIR / "parkinsons_clean.data", index=False)
    df_avg.to_csv(PROCESSED_DIR / "parkinsons_avg.data", index=False)
    df_norm.to_csv(PROCESSED_DIR / "parkinsons_norm.data", index=False)


def train_and_save_models(datasets: dict, final_features: list) -> str:
    exec_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = MODELS_DIR / exec_time
    logs_dir.mkdir(parents=True, exist_ok=True)
    logs_file_path = logs_dir / "metrics.txt"

    with open(logs_file_path, "w") as log_file:
        for model_name, dataset in datasets.items():
            x = dataset[final_features]
            y = dataset["status"]
            x = cast(pd.DataFrame, x)
            y = cast(pd.Series, y)

            model, _, metrics = model_generator(
                x, y, size=0.3, seed=123, n_neighbors=n_neighbors
            )

            model_path = logs_dir / f"{model_name}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            log_file.write(f"{model_name} with n_neighbors = {n_neighbors}\n")
            log_file.write(
                f"Metrics: Accuracy = {metrics['accuracy']:.2f}, "
                f"F1 Score = {metrics['f1_score']:.2f}, "
                f"Precision = {metrics['precision']:.2f}, "
                f"Recall = {metrics['recall']:.2f}\n\n"
            )

    return str(logs_dir)


def main():
    df_clean, df_avg, df_norm = load_and_process()
    save_processed_datasets(df_clean, df_avg, df_norm)

    final_features = select_features(df_norm)
    datasets = {"model_clean": df_clean, "model_avg": df_avg, "model_norm": df_norm}
    logs_dir = train_and_save_models(datasets, final_features)

    print(f"Models and metrics saved to: {logs_dir}")


if __name__ == "__main__":
    main()
