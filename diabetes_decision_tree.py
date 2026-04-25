# Diabetes Prediction - Decision Tree (Optimal: max_depth=7)

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

os.makedirs("model_outputs", exist_ok=True)

# Load cleaned dataset
df = pd.read_csv("diabetes_clean.csv")
X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model (depth=7 is the optimal - tested from depth 1 to 20)
model = DecisionTreeClassifier(max_depth=7, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
train_acc = accuracy_score(y_train, model.predict(X_train)) * 100
test_acc  = accuracy_score(y_test, y_pred) * 100
print(f"Train Accuracy : {train_acc:.2f}%")
print(f"Test  Accuracy : {test_acc:.2f}%")
print(f"Gap            : {train_acc - test_acc:.2f}%")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.title("Decision Tree - Confusion Matrix")
plt.colorbar()
plt.xticks([0, 1], ["No Diabetes", "Diabetes"])
plt.yticks([0, 1], ["No Diabetes", "Diabetes"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=14)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("model_outputs/dt_confusion_matrix.png")
plt.close()

# Feature importance plot
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
importance.plot(kind="bar", color="steelblue", edgecolor="white")
plt.title("Decision Tree - Feature Importance")
plt.ylabel("Importance Score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("model_outputs/dt_feature_importance.png")
plt.close()

print("Saved: dt_confusion_matrix.png, dt_feature_importance.png")
