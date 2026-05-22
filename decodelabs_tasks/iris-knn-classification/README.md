# Iris Data Classification with KNN

This project implements a complete data classification pipeline in Python using:
- Iris dataset (`sklearn.datasets.load_iris`)
- K-Nearest Neighbors (KNN)
- Feature scaling with `StandardScaler`
- Elbow method for selecting K
- Model evaluation metrics and confusion matrix visualization

## Files
- `knn_iris_classification.py`: Full pipeline implementation
- `requirements.txt`: Python dependencies

## Setup
1. Create and activate a virtual environment (recommended)
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run
```bash
python knn_iris_classification.py
```

The script prints exploration details and metrics to the console, and opens:
1. Error Rate vs K plot (elbow method)
2. Confusion matrix heatmap
