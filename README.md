<div align="justify">

# Parkinson's Disease Modeling API

This repository contains the complete workflow for modeling Parkinson’s disease
progression using K-Nearest Neighbors (KNN) and integrating the final model
into a minimal API for real-time prediction. The project includes code for data
processing, model training, and API deployment.

## 🚀 Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/RahulSandhu/parkinsons-project
   cd parkinsons-project
   ```

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## 🖥️ Source

* `src/api/`: FastAPI app and endpoint logic
* `src/main.py`: Entrypoint to generate models
* `src/model/`: KNN model training and feature selection
* `src/processing/`: Data preprocessing

## 📁 Data

* `data/raw/`: Original Parkinson’s disease dataset
* `data/processed/`: Cleaned and transformed data used for modeling
* `data/dummy/`: Synthetic examples used for testing the API

## 📊 Results

* `results/YYYYMMDD_HHMMSS/`: Timestamped directories containing results,
model files, and logs
* Best performance achieved by normalized KNN model with 97% accuracy at k = 4

## 🎓 Acknowledgements

* Dataset used for modeling Parkinson’s disease
* Developed as part of the Health Data Science Master’s program at Universitat
Rovira i Virgili (URV)

</div>
