import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Set working directory so all files save to the project folder
os.chdir(r"D:\Machine Learning\Project Diabetes Prediction")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve, roc_auc_score

# Load the dataset
df = pd.read_csv(r"D:\Machine Learning\Project Diabetes Prediction\diabetes_binary_5050_cleaned.csv")

print("Dataset shape:", df.shape)

# Separate features and target column
X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features - important for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred_proba)

print("Accuracy:", round(accuracy * 100, 2), "%")
print("AUC Score:", round(auc_score, 4))
print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))

# Graph: Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Diabetes", "Diabetes"])
disp.plot(cmap="Blues", colorbar=False)
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.savefig("confusion_matrix_lr.png", dpi=150)
plt.close()

# Graph: ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, lw=2, label=f"Logistic Regression (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curve_lr.png", dpi=150)
plt.close()

# Graph: Feature importance (coefficients)
coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

colors = ["#DD8452" if c > 0 else "#4C72B0" for c in coeff_df["Coefficient"]]
plt.figure(figsize=(9, 5))
plt.barh(coeff_df["Feature"], coeff_df["Coefficient"], color=colors)
plt.axvline(x=0, color="black", linewidth=0.8)
plt.xlabel("Coefficient Value")
plt.title("Feature Importance - Logistic Regression")
plt.tight_layout()
plt.savefig("feature_importance_lr.png", dpi=150)
plt.close()

# Save all test predictions to a CSV file
results_df = X_test.copy()
results_df["Actual"] = y_test.values
results_df["Predicted"] = y_pred
results_df["Probability"] = np.round(y_pred_proba, 4)
results_df["Actual_Label"] = results_df["Actual"].map({0: "No Diabetes", 1: "Diabetes"})
results_df["Predicted_Label"] = results_df["Predicted"].map({0: "No Diabetes", 1: "Diabetes"})
results_df["Correct?"] = (results_df["Actual"] == results_df["Predicted"]).map({True: "Yes", False: "No"})

print(results_df[["Actual_Label", "Predicted_Label", "Probability", "Correct?"]].head(10).to_string(index=False))

results_df[["Actual", "Actual_Label", "Predicted", "Predicted_Label", "Probability", "Correct?"]].to_csv("logistic_regression_predictions.csv", index=False)

# Predict diabetes for a new custom patient
patient_data = {
    "HighBP": 1,
    "HighChol": 1,
    "BMI": 32,
    "Smoker": 0,
    "Stroke": 0,
    "HeartDiseaseorAttack": 0,
    "PhysActivity": 1,
    "Fruits": 1,
    "Veggies": 1,
    "HvyAlcoholConsump": 0,
    "GenHlth": 3,
    "MentHlth": 2,
    "PhysHlth": 5,
    "DiffWalk": 0,
    "Sex": 1,
    "Age": 9
}

patient_df = pd.DataFrame([patient_data])
patient_scaled = scaler.transform(patient_df)

pred_label = model.predict(patient_scaled)[0]
pred_proba = model.predict_proba(patient_scaled)[0][1]

print("\nCustom Patient Prediction:")
if pred_label == 1:
    print("Result: DIABETES DETECTED")
else:
    print("Result: NO DIABETES")
print("Probability:", round(pred_proba * 100, 2), "%")

print("\nDone! Check the saved PNG files and logistic_regression_predictions.csv")
