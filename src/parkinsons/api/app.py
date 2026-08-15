import os
import pickle

import pandas as pd
import uvicorn
from fastapi import FastAPI, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse

from parkinsons.utils.config import MODELS_DIR, PROCESSED_DIR, STATIC_DIR

app = FastAPI()

data_path = PROCESSED_DIR / "parkinsons_clean.data"
data_df = pd.read_csv(data_path)
feature_max_min = {
    feature: (data_df[feature].max(), data_df[feature].min())
    for feature in data_df.columns
    if data_df[feature].dtype in ["float64", "int64"]
}


@app.get("/", response_class=HTMLResponse)
def read_root() -> HTMLResponse:
    with open(STATIC_DIR / "index.html") as f:
        return HTMLResponse(f.read())


@app.get("/datetimes/")
def get_datetimes() -> JSONResponse:
    datetime_folders = [
        folder
        for folder in os.listdir(MODELS_DIR)
        if os.path.isdir(os.path.join(MODELS_DIR, folder))
    ]
    return JSONResponse(content=datetime_folders)


@app.get("/models/")
def get_models(datetime_folder: str = Query(...)) -> JSONResponse:
    folder_path = os.path.join(MODELS_DIR, datetime_folder)
    models = [file for file in os.listdir(folder_path) if file.endswith(".pkl")]
    return JSONResponse(content=models)


@app.get("/metrics/")
def get_metrics(datetime_folder: str = Query(...)) -> str:
    folder_path = os.path.join(MODELS_DIR, datetime_folder)
    metrics_path = os.path.join(folder_path, "metrics.txt")
    with open(metrics_path) as f:
        content = f.read()
        return content.replace("\n", "<br>")


@app.post("/predict/")
def predict(  # noqa: PLR0913
    datetime_folder: str = Form(...),
    model_name: str = Form(...),
    avFF: float = Form(None),  # noqa: N803
    maxFF: float = Form(None),  # noqa: N803
    minFF: float = Form(None),  # noqa: N803
    percJitter: float = Form(None),  # noqa: N803
    absJitter: float = Form(None),  # noqa: N803
    rap: float = Form(None),
    ppq: float = Form(None),
    ddp: float = Form(None),
    lShimmer: float = Form(None),  # noqa: N803
    dbShimmer: float = Form(None),  # noqa: N803
    apq3: float = Form(None),
    apq5: float = Form(None),
    apq: float = Form(None),
    dda: float = Form(None),
    NHR: float = Form(None),  # noqa: N803
    HNR: float = Form(None),  # noqa: N803
    RPDE: float = Form(None),  # noqa: N803
    DFA: float = Form(None),  # noqa: N803
    spread1: float = Form(None),
    spread2: float = Form(None),
    D2: float = Form(None),  # noqa: N803
    PPE: float = Form(None),  # noqa: N803
) -> dict:
    folder_path = os.path.join(MODELS_DIR, datetime_folder)
    model_path = os.path.join(folder_path, model_name)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    required_features = (
        model.feature_names_in_.tolist() if hasattr(model, "feature_names_in_") else []
    )

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

    if model_name == "model_norm.pkl":
        for feature in required_features:
            if feature in all_features and all_features[feature] is not None:
                max_value, min_value = feature_max_min.get(feature, (None, None))
                if max_value is not None and min_value is not None:
                    all_features[feature] = (all_features[feature] - min_value) / (
                        max_value - min_value
                    )

    input_df = pd.DataFrame(
        [{feature: all_features[feature] for feature in required_features}]
    )
    prediction = int(model.predict(input_df)[0])

    return {
        "datetime_folder": datetime_folder,
        "model": model_name,
        "prediction": prediction,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
