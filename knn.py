import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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

# Scale the features - important for KNN because it uses distance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Find the best K by testing K from 1 to 20
k_values = range(1, 21)
train_scores = []
test_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    train_scores.append(accuracy_score(y_train, knn.predict(X_train_scaled)))
    test_scores.append(accuracy_score(y_test, knn.predict(X_test_scaled)))

best_k = k_values[np.argmax(test_scores)]
print("Best K:", best_k)

# Graph: Train vs Test accuracy for each K value to find the best K
plt.figure(figsize=(10, 5))
plt.plot(k_values, [s * 100 for s in train_scores], marker="o", label="Train Accuracy")
plt.plot(k_values, [s * 100 for s in test_scores], marker="s", label="Test Accuracy")
plt.axvline(x=best_k, color="red", linestyle="--", label=f"Best K = {best_k}")
plt.xlabel("K Value")
plt.ylabel("Accuracy (%)")
plt.title("KNN - Accuracy vs K Value")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("knn_best_k.png", dpi=150)
plt.close()

# Train the final KNN model using the best K
model_knn = KNeighborsClassifier(n_neighbors=best_k)
model_knn.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model_knn.predict(X_test_scaled)
y_pred_proba = model_knn.predict_proba(X_test_scaled)[:, 1]

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred_proba)

print("Accuracy:", round(accuracy * 100, 2), "%")
print("AUC Score:", round(auc_score, 4))
print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))

# Graph: Confusion matrix showing correct and wrong predictions
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Diabetes", "Diabetes"])
disp.plot(cmap="Oranges", colorbar=False)
plt.title(f"Confusion Matrix - KNN (K={best_k})")
plt.tight_layout()
plt.savefig("confusion_matrix_knn.png", dpi=150)
plt.close()

# Graph: ROC curve showing how well the model separates the two classes
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, lw=2, label=f"KNN (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - KNN (K={best_k})")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curve_knn.png", dpi=150)
plt.close()

# Cross validation to check if the model is consistent across different splits
cv_scores = cross_val_score(KNeighborsClassifier(n_neighbors=best_k), X_train_scaled, y_train, cv=5, scoring="accuracy")
print("CV Scores:", [round(s * 100, 2) for s in cv_scores])
print("Mean CV Accuracy:", round(cv_scores.mean() * 100, 2), "%")

# Compare KNN with Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test_scaled))
lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1])

print("\nModel Comparison:")
print("Logistic Regression - Accuracy:", round(lr_acc * 100, 2), "%  AUC:", round(lr_auc, 4))
print(f"KNN (K={best_k})         - Accuracy:", round(accuracy * 100, 2), "%  AUC:", round(auc_score, 4))

models = ["Logistic\nRegression", f"KNN\n(K={best_k})"]
acc_vals = [lr_acc * 100, accuracy * 100]
auc_vals = [lr_auc, auc_score]

# Graph: Side by side bar chart comparing Accuracy and AUC of both models
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("Logistic Regression vs KNN", fontsize=14)

bars1 = axes[0].bar(models, acc_vals, color=["#4C72B0", "#DD8452"], width=0.4)
axes[0].set_title("Accuracy (%)")
axes[0].set_ylim(60, 85)
for bar, val in zip(bars1, acc_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"{val:.2f}%", ha="center", fontweight="bold")

bars2 = axes[1].bar(models, auc_vals, color=["#4C72B0", "#DD8452"], width=0.4)
axes[1].set_title("AUC Score")
axes[1].set_ylim(0.6, 0.95)
for bar, val in zip(bars2, auc_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, f"{val:.4f}", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("lr_vs_knn_comparison.png", dpi=150)
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

results_df[["Actual", "Actual_Label", "Predicted", "Predicted_Label", "Probability", "Correct?"]].to_csv("knn_predictions.csv", index=False)

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

pred_label = model_knn.predict(patient_scaled)[0]
pred_proba = model_knn.predict_proba(patient_scaled)[0][1]

print("\nCustom Patient Prediction:")
if pred_label == 1:
    print("Result: DIABETES DETECTED")
else:
    print("Result: NO DIABETES")
print("Probability:", round(pred_proba * 100, 2), "%")

print("\nDone! Check the saved PNG files and knn_predictions.csv")
