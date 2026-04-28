# ===============================
# 1. Import Libraries
# ===============================
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import roc_curve, roc_auc_score

# ===============================
# 2. Load & Clean Data
# ===============================
df = pd.read_csv(r"E:\machine learning model\final dataset.csv.xls")

df = df.drop_duplicates()
df = df[(df['BMI'] >= 15) & (df['BMI'] <= 60)]
df = df.dropna()

X = df.drop('Diabetes_binary', axis=1)
y = df['Diabetes_binary']

# ===============================
# 3. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 4. Scaling (ONLY for some models)
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# 5. Initialize Models
# ===============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel='rbf', C=10, gamma=0.01, probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(max_depth=6),
    "Random Forest": RandomForestClassifier(n_estimators=200),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

# ===============================
# 6. Train + ROC Calculation
# ===============================
plt.figure(figsize=(8,6))

for name, model in models.items():
    
    # Use scaled data for these models
    if name in ["Logistic Regression", "SVM", "KNN"]:
        model.fit(X_train_scaled, y_train)
        y_prob = model.predict_proba(X_test_scaled)[:,1]
    else:
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:,1]
    
    # ROC values
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})")

# ===============================
# 7. Plot
# ===============================
plt.plot([0,1], [0,1], linestyle='--')  # diagonal
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison of Models")
plt.legend()
plt.show()

import pandas as pd
from sklearn.metrics import roc_curve

# Store results
roc_data = {}

models = {
    "Logistic Regression": log_model,
    "SVM": svm_model,
    "KNN": knn_model,
    "Decision Tree": dt_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

for name, model in models.items():
    
    # Use scaled data for some models
    if name in ["Logistic Regression", "SVM", "KNN"]:
        y_prob = model.predict_proba(X_test_scaled)[:,1]
    else:
        y_prob = model.predict_proba(X_test)[:,1]
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    
    # Store TPR indexed by FPR
    roc_data[name] = pd.Series(tpr, index=fpr)

# ===============================
# Combine into one table
# ===============================
roc_df = pd.DataFrame(roc_data)

# Fill missing values
roc_df = roc_df.interpolate().fillna(method='bfill').fillna(method='ffill')

# Reset index for table format
roc_df.reset_index(inplace=True)
roc_df.rename(columns={"index": "FPR"}, inplace=True)

# ===============================
# Show table
# ===============================
print(roc_df.head(10))

# Save to CSV (optional)
roc_df.to_csv("roc_comparison_table.csv", index=False)