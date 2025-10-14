import os
import pickle
from datetime import datetime
from typing import cast

import pandas as pd

from model.collinearity import collinearity
from model.model_generator import model_generator
from processing.aggregate import aggregate
from processing.normalize import normalize
from processing.outliers import outliers
from processing.rename import rename

# Load dataset
df = pd.read_csv("../data/raw/parkinsons.data")

# Rename columns
df_rename = rename(df)

# Remove outliers
df_clean = outliers(df_rename)

# Aggregate by subject_id
df_avg = aggregate(df_clean, "subject_id")

# Normalize DataFrame
df_norm = normalize(df_clean)

# Define feature groups
feature_groups = {
    "Fundamental Frequency": ["avFF", "maxFF", "minFF"],
    "Jitter": ["absJitter", "percJitter", "rap", "ppq", "ddp"],
    "Shimmer": ["lShimmer", "dbShimmer", "apq3", "apq5", "apq", "dda"],
}

# Assess collinearity and get features to remove
remove_features = collinearity(df_norm, feature_groups)
final_features = set(df_norm.columns) - set(remove_features)
final_features = list(final_features)
final_features.remove("subject_id")
final_features.remove("trial")
final_features.remove("status")

# Define base directory logs
base_dir = "../results/"
exec_time = datetime.now().strftime("%Y%m%d_%H%M%S")
logs_dir = os.path.join(base_dir, exec_time)
os.makedirs(logs_dir, exist_ok=True)

# Define logs file path
logs_file_path = os.path.join(logs_dir, "metrics.txt")

# Number of neighbors
n_neighbors = 5

# DataFrames to process
datasets = {"model_clean": df_clean, "model_avg": df_avg, "model_norm": df_norm}

# Generate, save models, and output metrics
with open(logs_file_path, "w") as log_file:
    for model_name, dataset in datasets.items():
        X = dataset[final_features]
        Y = dataset["status"]

        # Define model path
        model_path = os.path.join(logs_dir, f"{model_name}.pkl")

        # Fix for PyRight
        X = cast(pd.DataFrame, X)
        Y = cast(pd.Series, Y)

        # Generate and save model
        model, _, metrics = model_generator(
            X,
            Y,
            size=0.3,
            seed=123,
            n_neighbors=n_neighbors,
        )
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Log results to the log file
        log_file.write(f"{model_name} with n_neighbors = {n_neighbors}\n")
        log_file.write(
            f"Metrics: Accuracy = {metrics['accuracy']:.2f}, "
            f"F1 Score = {metrics['f1_score']:.2f}, Precision = {metrics['precision']:.2f}, "
            f"Recall = {metrics['recall']:.2f}\n\n"
        )
