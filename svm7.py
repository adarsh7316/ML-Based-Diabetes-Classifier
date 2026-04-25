# ===============================
# 1. Import Libraries
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

# ===============================
# 2. Load Dataset
# ===============================
df = pd.read_csv(r"E:\machine learning model\final dataset.csv.xls")

# ===============================
# 3. Define Features & Target
# ===============================
X = df.drop('Diabetes_binary', axis=1)
y = df['Diabetes_binary']

# ===============================
# 4. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 5. Feature Scaling
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# 6. Train SVM Model
# ===============================
model = SVC(kernel='rbf', C=10, gamma=0.01, probability=True)
model.fit(X_train, y_train)

# ===============================
# 7. Predictions
# ===============================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ===============================
# 8. Evaluation Metrics
# ===============================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("===== Model Performance =====")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# ===============================
# 9. Confusion Matrix + Rates
# ===============================
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n===== Confusion Matrix Values =====")
print("True Positive (TP):", tp)
print("True Negative (TN):", tn)
print("False Positive (FP):", fp)
print("False Negative (FN):", fn)

# 🔥 Rates
tpr = tp / (tp + fn)   # Recall / Sensitivity
fpr = fp / (fp + tn)
tnr = tn / (tn + fp)   # Specificity
fnr = fn / (fn + tp)

print("\n===== Rates =====")
print("TPR (Recall / Sensitivity):", tpr)
print("FPR:", fpr)
print("TNR (Specificity):", tnr)
print("FNR:", fnr)

# ===============================
# 10. Confusion Matrix Plot
# ===============================
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["No Diabetes", "Diabetes"],
            yticklabels=["No Diabetes", "Diabetes"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SVM")
plt.show()

# ===============================
# 11. ROC Curve & AUC
# ===============================
auc = roc_auc_score(y_test, y_prob)
print("\nAUC Score:", auc)

fpr_curve, tpr_curve, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr_curve, tpr_curve, label="SVM (AUC = %.2f)" % auc)
plt.plot([0,1], [0,1], 'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - SVM")
plt.legend()
plt.show()