# 🩺 Diabetes Prediction Project

A machine learning project that predicts diabetes using multiple classification models. Built as part of a second-year machine learning course.

## 📂 Project Structure

```
Project Diabetes Prediction/
│
├── logistic_regression.py       # Logistic Regression model
├── knn.py                       # K-Nearest Neighbors model
│
├── confusion_matrix_lr.png      # Confusion matrix - Logistic Regression
├── confusion_matrix_knn.png     # Confusion matrix - KNN
├── roc_curve_lr.png             # ROC Curve - Logistic Regression
├── roc_curve_knn.png            # ROC Curve - KNN
├── feature_importance_lr.png    # Feature Importance - Logistic Regression
├── knn_best_k.png               # Best K value plot - KNN
├── lr_vs_knn_comparison.png     # Model Comparison Chart
│
└── README.md
```

## 🤖 Models Implemented

| Model | Description |
|---|---|
| Logistic Regression | Binary classification using logistic function |
| K-Nearest Neighbors (KNN) | Instance-based learning with optimal K selection |

## 📊 Dataset

- **Source**: CDC Diabetes Health Indicators Dataset (Binary, 50/50 balanced)
- **Rows**: ~63,430 records
- **Features**: 17 health-related features

## 📈 Evaluation Metrics

- Accuracy
- AUC-ROC Score
- Confusion Matrix
- Feature Importance (Logistic Regression)

## 🛠️ Libraries Used

- `pandas` — Data manipulation
- `numpy` — Numerical operations
- `scikit-learn` — ML models and evaluation
- `matplotlib` / `seaborn` — Visualizations

## 🚀 How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Run Logistic Regression
python logistic_regression.py

# Run KNN
python knn.py
```

## 👤 Author

Second-year Machine Learning Student
