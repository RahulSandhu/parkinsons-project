<div align="justify">

# Parkinson's Disease Modeling API

This repository contains the complete workflow for modeling Parkinson's disease
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

## 📁 Dataset

The project analyzes a dataset of Parkinson's disease patients with comprehensive
voice measurement features. The dataset contains biomedical voice measurements
from individuals, with key features including:

- **Voice frequency measures**: Average vocal fundamental frequency, variation
  in fundamental frequency
- **Amplitude measures**: Variations in amplitude, noise-to-tonal component
  ratios
- **Harmonic measures**: Harmonic-to-noise ratio, correlation measures
- **Nonlinear measures**: Recurrence period density entropy, detrended
  fluctuation analysis
- **Signal complexity**: Pitch period entropy, fundamental frequency variation

The analysis uses **K-Nearest Neighbors (KNN)** with feature selection and
normalization to predict Parkinson's disease status.

## 📊 Results

- Best performance achieved by normalized KNN model with **97% accuracy** at
  k = 4
- Feature selection and normalization significantly improved model performance
- Model integrated into FastAPI for real-time predictions
- Comprehensive evaluation metrics including precision, recall, and F1-score
  demonstrate robust classification performance

## 🎓 Acknowledgements

- [Kaggle](https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set)
  – Parkinson's Disease dataset
- Developed as part of the Scientific Programming course in the Master in
  Health Data Science program at Universitat Rovira i Virgili (URV)

</div>
