import os
import pickle

import pandas as pd
import uvicorn
from fastapi import FastAPI, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse

# Directory for model files
MODEL_DIR = "../../results/"

# Initialize API
app = FastAPI()

# Load processed Parkinson's dataset for normalization reference
data_path = "../../data/processed/parkinsons_clean.data"
data_df = pd.read_csv(data_path)
feature_max_min = {
    feature: (data_df[feature].max(), data_df[feature].min())
    for feature in data_df.columns
    if data_df[feature].dtype in ["float64", "int64"]
}


@app.get("/", response_class=HTMLResponse)
def read_root() -> HTMLResponse:
    """
    Use 'index.html' as HTML page.

    Inputs:
        - None.

    Outputs:
        - HTMLResponse: The content of 'index.html' served as an HTML page.
    """
    with open("index.html", "r") as f:
        return HTMLResponse(f.read())


@app.get("/datetimes/")
def get_datetimes() -> JSONResponse:
    """
    Retrieve a list of datetime-named folders in the model directory.

    Inputs:
        - None.

    Outputs:
        - JSONResponse: A JSON list of folder names in the model directory.
    """
    datetime_folders = [
        folder
        for folder in os.listdir(MODEL_DIR)
        if os.path.isdir(os.path.join(MODEL_DIR, folder))
    ]
    return JSONResponse(content=datetime_folders)


@app.get("/models/")
def get_models(datetime_folder: str = Query(...)) -> JSONResponse:
    """
    Retrieve a list of model files in a specified datetime folder.

    Inputs:
        - datetime_folder (str): The name of the datetime folder to search for
          models.

    Outputs:
        - JSONResponse: A JSON list of model file names ending with '.pkl'.
    """
    folder_path = os.path.join(MODEL_DIR, datetime_folder)
    models = [file for file in os.listdir(folder_path) if file.endswith(".pkl")]
    return JSONResponse(content=models)


@app.get("/metrics/")
def get_metrics(datetime_folder: str = Query(...)) -> str:
    """
    Retrieve the content of the 'metrics.txt' file in a specified datetime
    folder.

    Inputs:
        - datetime_folder (str): The name of the datetime folder containing the
          metrics file.

    Outputs:
        - str: The content of the 'metrics.txt' file with line breaks properly
          formatted.
    """
    folder_path = os.path.join(MODEL_DIR, datetime_folder)
    metrics_path = os.path.join(folder_path, "metrics.txt")
    with open(metrics_path, "r") as f:
        content = f.read()
        content = content.replace("\n", "<br>")
        return content


@app.post("/predict/")
def predict(
    datetime_folder: str = Form(...),
    model_name: str = Form(...),
    avFF: float = Form(None),
    maxFF: float = Form(None),
    minFF: float = Form(None),
    percJitter: float = Form(None),
    absJitter: float = Form(None),
    rap: float = Form(None),
    ppq: float = Form(None),
    ddp: float = Form(None),
    lShimmer: float = Form(None),
    dbShimmer: float = Form(None),
    apq3: float = Form(None),
    apq5: float = Form(None),
    apq: float = Form(None),
    dda: float = Form(None),
    NHR: float = Form(None),
    HNR: float = Form(None),
    RPDE: float = Form(None),
    DFA: float = Form(None),
    spread1: float = Form(None),
    spread2: float = Form(None),
    D2: float = Form(None),
    PPE: float = Form(None),
) -> dict:
    """
    Make a prediction using a specified model and input features.

    Inputs:
        - datetime_folder (str): The name of the folder containing the model.
        - model_name (str): The name of the model file to use for prediction.
        - avFF, maxFF, minFF, percJitter, absJitter, rap, ppq, ddp, lShimmer,
          dbShimmer, apq3, apq5, apq, dda, NHR, HNR, RPDE, DFA, spread1,
          spread2, D2, PPE (float, optional): Feature values for prediction.

    Outputs:
        - dict: A dictionary containing the folder name, model name, and
          prediction result.
    """
    # Get folder and model path
    folder_path = os.path.join(MODEL_DIR, datetime_folder)
    model_path = os.path.join(folder_path, model_name)

    # Load the model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Select needed features
    required_features = (
        model.feature_names_in_.tolist() if hasattr(model, "feature_names_in_") else []
    )

    # Collect input features
    all_features = {
        "avFF": avFF,
        "maxFF": maxFF,
        "minFF": minFF,
        "percJitter": percJitter,
        "absJitter": absJitter,
        "rap": rap,
        "ppq": ppq,
        "ddp": ddp,
        "lShimmer": lShimmer,
        "dbShimmer": dbShimmer,
        "apq3": apq3,
        "apq5": apq5,
        "apq": apq,
        "dda": dda,
        "NHR": NHR,
        "HNR": HNR,
        "RPDE": RPDE,
        "DFA": DFA,
        "spread1": spread1,
        "spread2": spread2,
        "D2": D2,
        "PPE": PPE,
    }

    # Normalize inputs if the model requires normalization
    if model_name == "model_norm.pkl":
        for feature in required_features:
            if feature in all_features and all_features[feature] is not None:
                max_value, min_value = feature_max_min.get(feature, (None, None))

                if max_value is not None and min_value is not None:
                    all_features[feature] = (all_features[feature] - min_value) / (
                        max_value - min_value
                    )

    # Prediction
    input_df = pd.DataFrame(
        [{feature: all_features[feature] for feature in required_features}]
    )
    prediction = model.predict(input_df)[0]
    prediction = int(prediction)

    return {
        "datetime_folder": datetime_folder,
        "model": model_name,
        "prediction": prediction,
    }


if __name__ == "__main__":
    # Launch API
    uvicorn.run(app, host="0.0.0.0", port=8000)
