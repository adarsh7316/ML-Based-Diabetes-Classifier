# Diabetes EDA - Exploratory Data Analysis
# Dataset: diabetes_binary_5050split_health_indicators_BRFSS2015.csv
# Pipeline: Load → Data Cleaning → Outlier Removal → EDA

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # saves plots to files instead of opening a window
import matplotlib.pyplot as plt
import seaborn as sns

# ================================================================
# STEP 1: Load the raw dataset
# ================================================================

df_raw = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

print("=" * 65)
print("STEP 1 — RAW DATASET")
print("=" * 65)
print(f"Shape (rows, columns): {df_raw.shape}")
print("\nFirst 5 rows:")
print(df_raw.head())
print("\nColumn names:")
print(df_raw.columns.tolist())
print("\nData types:")
print(df_raw.dtypes)

# ================================================================
# STEP 2: DATA CLEANING
# ================================================================

print("\n" + "=" * 65)
print("STEP 2 — DATA CLEANING")
print("=" * 65)

df = df_raw.copy()

# ------------------------------------------------------------------
# 2a. Missing values
# ------------------------------------------------------------------
print("\n[2a] Missing values per column:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "  ✅ No missing values found.")

# Drop rows with any missing values (safety net)
rows_before = len(df)
df.dropna(inplace=True)
print(f"  Rows dropped due to missing values: {rows_before - len(df)}")

# ------------------------------------------------------------------
# 2b. Duplicate rows
# ------------------------------------------------------------------
dupes = df.duplicated().sum()
print(f"\n[2b] Duplicate rows found: {dupes}")
df.drop_duplicates(inplace=True)
print(f"  Rows after removing duplicates: {len(df)}")

# ------------------------------------------------------------------
# 2c. Validate binary columns (must be 0 or 1)
# ------------------------------------------------------------------
binary_cols = [
    "Diabetes_binary", "HighBP", "HighChol", "CholCheck", "Smoker",
    "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits",
    "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
    "DiffWalk", "Sex"
]

print("\n[2c] Validating binary columns (valid values: 0 or 1):")
invalid_binary_mask = pd.Series([False] * len(df), index=df.index)
for col in binary_cols:
    invalid = ~df[col].isin([0, 1])
    count = invalid.sum()
    if count > 0:
        print(f"  ⚠️  {col}: {count} invalid values → removing those rows")
        invalid_binary_mask |= invalid
    else:
        print(f"  ✅ {col}: OK")

rows_before = len(df)
df = df[~invalid_binary_mask]
print(f"  Rows removed for invalid binary values: {rows_before - len(df)}")

# ------------------------------------------------------------------
# 2d. Validate ordinal / range columns
# ------------------------------------------------------------------
ordinal_ranges = {
    "GenHlth":   (1, 5),
    "MentHlth":  (0, 30),
    "PhysHlth":  (0, 30),
    "Age":       (1, 13),
    "Education": (1, 6),
    "Income":    (1, 8),
}

print("\n[2d] Validating ordinal/range columns:")
invalid_ordinal_mask = pd.Series([False] * len(df), index=df.index)
for col, (lo, hi) in ordinal_ranges.items():
    invalid = (df[col] < lo) | (df[col] > hi)
    count = invalid.sum()
    if count > 0:
        print(f"  ⚠️  {col}: {count} values outside [{lo}, {hi}] → removing")
        invalid_ordinal_mask |= invalid
    else:
        print(f"  ✅ {col}: all values in [{lo}, {hi}]")

rows_before = len(df)
df = df[~invalid_ordinal_mask]
print(f"  Rows removed for out-of-range ordinal values: {rows_before - len(df)}")

# ------------------------------------------------------------------
# 2e. Validate BMI (reasonable human range: 10 – 80)
# ------------------------------------------------------------------
print("\n[2e] Validating BMI (expected range: 10–80):")
bmi_invalid = (df["BMI"] < 10) | (df["BMI"] > 80)
count = bmi_invalid.sum()
if count > 0:
    print(f"  ⚠️  BMI: {count} physiologically implausible values → removing")
    df = df[~bmi_invalid]
else:
    print(f"  ✅ BMI: all values in [10, 80]")

# ------------------------------------------------------------------
# 2f. Ensure correct dtypes (all should be numeric)
# ------------------------------------------------------------------
print("\n[2f] Ensuring numeric dtypes:")
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
remaining_na = df.isnull().sum().sum()
if remaining_na > 0:
    print(f"  ⚠️  {remaining_na} cells became NaN after coercion → dropping")
    df.dropna(inplace=True)
else:
    print("  ✅ All columns are numeric, no coercion issues.")

print(f"\n📋 Dataset shape after cleaning: {df.shape}")

# ================================================================
# STEP 3: OUTLIER REMOVAL (IQR method on continuous columns)
# ================================================================

print("\n" + "=" * 65)
print("STEP 3 — OUTLIER REMOVAL (IQR Method)")
print("=" * 65)

# Only continuous columns are candidates for IQR-based outlier removal.
# Binary and ordinal columns were already validated above.
continuous_cols = ["BMI", "MentHlth", "PhysHlth"]

print("\nOutlier statistics BEFORE removal:")
print(f"  {'Column':<12} {'Q1':>8} {'Q3':>8} {'IQR':>8} {'Lower':>8} {'Upper':>8} {'Outliers':>10}")
print("  " + "-" * 60)

outlier_mask = pd.Series([False] * len(df), index=df.index)

for col in continuous_cols:
    Q1    = df[col].quantile(0.25)
    Q3    = df[col].quantile(0.75)
    IQR   = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    col_outliers = (df[col] < lower) | (df[col] > upper)
    n_out = col_outliers.sum()
    outlier_mask |= col_outliers
    print(f"  {col:<12} {Q1:>8.2f} {Q3:>8.2f} {IQR:>8.2f} {lower:>8.2f} {upper:>8.2f} {n_out:>10,}")

rows_before = len(df)
df = df[~outlier_mask]
rows_removed = rows_before - len(df)
print(f"\n  Total rows removed as outliers : {rows_removed:,}")
print(f"  Rows remaining (clean dataset) : {len(df):,}")

# ------------------------------------------------------------------
# Boxplot: before vs after is shown by plotting on the clean df
# ------------------------------------------------------------------
plt.figure(figsize=(12, 4))
for i, col in enumerate(continuous_cols):
    plt.subplot(1, 3, i + 1)
    plt.boxplot(df[col], patch_artist=True,
                boxprops=dict(facecolor="steelblue", alpha=0.6))
    plt.title(f"{col}\n(after outlier removal)")
    plt.ylabel(col)

plt.suptitle("Continuous Features – After Outlier Removal", fontsize=14)
plt.tight_layout()
plt.savefig("03_outlier_removed_boxplots.png")
plt.close()
print("\nSaved: 03_outlier_removed_boxplots.png")

# ================================================================
# STEP 4: Save the clean dataset
# ================================================================

clean_path = "diabetes_clean.csv"
df.to_csv(clean_path, index=False)
print(f"\n✅ Clean dataset saved → '{clean_path}'  |  Shape: {df.shape}")

# ================================================================
# EDA ON CLEAN DATASET
# ================================================================

print("\n" + "=" * 65)
print("EDA — EXPLORATORY DATA ANALYSIS ON CLEAN DATASET")
print("=" * 65)
print(f"\nClean dataset shape: {df.shape}")
print("\nBasic statistics:")
print(df.describe())

# ------------------------------------------------------------------
# STEP 5: Target column distribution (class balance)
# ------------------------------------------------------------------

print("\nTarget column value counts:")
print(df["Diabetes_binary"].value_counts())

plt.figure(figsize=(6, 4))
df["Diabetes_binary"].value_counts().plot(
    kind="bar", color=["steelblue", "salmon"]
)
plt.title("Diabetes vs No Diabetes – Clean Dataset")
plt.xlabel("0 = No Diabetes,  1 = Diabetes")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("04_class_distribution.png")
plt.close()
print("Saved: 04_class_distribution.png")

# ------------------------------------------------------------------
# STEP 6: Distribution of binary features
# ------------------------------------------------------------------

binary_feat_cols = [c for c in binary_cols if c != "Diabetes_binary"]

plt.figure(figsize=(18, 14))
for i, col in enumerate(binary_feat_cols):
    plt.subplot(4, 4, i + 1)
    df[col].value_counts().plot(kind="bar", color=["steelblue", "salmon"])
    plt.title(col)
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.xticks(rotation=0)

plt.suptitle("Distribution of Binary Features (Clean Data)", fontsize=14)
plt.tight_layout()
plt.savefig("05_binary_features.png")
plt.close()
print("Saved: 05_binary_features.png")

# ------------------------------------------------------------------
# STEP 7: Distribution of continuous & ordinal features
# ------------------------------------------------------------------

ordinal_cols = ["GenHlth", "Age", "Education", "Income"]

# Histograms for continuous columns
plt.figure(figsize=(14, 4))
for i, col in enumerate(continuous_cols):
    plt.subplot(1, 3, i + 1)
    plt.hist(df[col], bins=30, color="steelblue", edgecolor="white")
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel("Count")

plt.suptitle("Distribution of Continuous Features (Clean Data)", fontsize=14)
plt.tight_layout()
plt.savefig("06_continuous_features.png")
plt.close()
print("Saved: 06_continuous_features.png")

# Bar charts for ordinal columns
plt.figure(figsize=(18, 4))
for i, col in enumerate(ordinal_cols):
    plt.subplot(1, 4, i + 1)
    df[col].value_counts().sort_index().plot(
        kind="bar", color="steelblue", edgecolor="white"
    )
    plt.title(col)
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=0)

plt.suptitle("Distribution of Ordinal Features (Clean Data)", fontsize=14)
plt.tight_layout()
plt.savefig("06b_ordinal_features.png")
plt.close()
print("Saved: 06b_ordinal_features.png")

# ------------------------------------------------------------------
# STEP 8: Compare each feature with the target (Diabetes)
# ------------------------------------------------------------------

# Binary features: % of each group that has diabetes
plt.figure(figsize=(18, 14))
for i, col in enumerate(binary_feat_cols):
    plt.subplot(4, 4, i + 1)
    group = df.groupby(col)["Diabetes_binary"].mean() * 100
    group.plot(kind="bar", color=["steelblue", "salmon"])
    plt.title(col)
    plt.ylabel("% with Diabetes")
    plt.xlabel("")
    plt.xticks(rotation=0)

plt.suptitle("% of Diabetics in each Binary Feature Group", fontsize=14)
plt.tight_layout()
plt.savefig("07_binary_vs_target.png")
plt.close()
print("Saved: 07_binary_vs_target.png")

# Continuous features: boxplot grouped by target
plt.figure(figsize=(14, 4))
for i, col in enumerate(continuous_cols):
    plt.subplot(1, 3, i + 1)
    group0 = df[df["Diabetes_binary"] == 0][col]
    group1 = df[df["Diabetes_binary"] == 1][col]
    plt.boxplot([group0, group1],
                labels=["No Diabetes", "Diabetes"],
                patch_artist=True)
    plt.title(col)
    plt.ylabel(col)

plt.suptitle("Continuous Features vs Target (Boxplot)", fontsize=14)
plt.tight_layout()
plt.savefig("07b_continuous_vs_target.png")
plt.close()
print("Saved: 07b_continuous_vs_target.png")

# Ordinal features: % with diabetes per category
plt.figure(figsize=(18, 4))
for i, col in enumerate(ordinal_cols):
    plt.subplot(1, 4, i + 1)
    group = df.groupby(col)["Diabetes_binary"].mean() * 100
    group.plot(kind="bar", color="steelblue", edgecolor="white")
    plt.title(col)
    plt.ylabel("% with Diabetes")
    plt.xticks(rotation=0)

plt.suptitle("% of Diabetics per Ordinal Feature Category", fontsize=14)
plt.tight_layout()
plt.savefig("07c_ordinal_vs_target.png")
plt.close()
print("Saved: 07c_ordinal_vs_target.png")

# ------------------------------------------------------------------
# STEP 9: Outlier check on clean data (confirm clean)
# ------------------------------------------------------------------

plt.figure(figsize=(12, 4))
for i, col in enumerate(continuous_cols):
    plt.subplot(1, 3, i + 1)
    plt.boxplot(df[col], patch_artist=True)
    plt.title(col)
    plt.ylabel(col)

plt.suptitle("Post-Cleaning Outlier Check – Continuous Features", fontsize=14)
plt.tight_layout()
plt.savefig("08_outliers_final.png")
plt.close()
print("Saved: 08_outliers_final.png")

print("\nPost-cleaning outlier counts (IQR method) — should all be 0:")
for col in continuous_cols:
    Q1    = df[col].quantile(0.25)
    Q3    = df[col].quantile(0.75)
    IQR   = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"  {col}: {n_out} outliers remaining")

# ------------------------------------------------------------------
# STEP 10: Correlation heatmap
# ------------------------------------------------------------------

plt.figure(figsize=(16, 12))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap – Clean Dataset")
plt.tight_layout()
plt.savefig("09_correlation_heatmap.png")
plt.close()
print("Saved: 09_correlation_heatmap.png")

# Features most correlated with Diabetes
print("\nCorrelation of each feature with Diabetes_binary:")
target_corr = (
    corr_matrix["Diabetes_binary"]
    .drop("Diabetes_binary")
    .sort_values(ascending=False)
)
print(target_corr)

plt.figure(figsize=(10, 6))
colors = ["salmon" if v > 0 else "steelblue" for v in target_corr.values]
plt.barh(target_corr.index, target_corr.values, color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Feature Correlation with Diabetes_binary (Clean Data)")
plt.xlabel("Pearson Correlation")
plt.tight_layout()
plt.savefig("09b_feature_correlations.png")
plt.close()
print("Saved: 09b_feature_correlations.png")

# ================================================================
# Summary
# ================================================================

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  Raw dataset rows        : {len(df_raw):,}")
print(f"  Clean dataset rows      : {len(df):,}")
print(f"  Rows removed (total)    : {len(df_raw) - len(df):,}")
print(f"  Clean file saved at     : {clean_path}")
print(f"  EDA plots saved in      : current directory (*.png)")
print("\n✅ Done! Data cleaning, outlier removal, and EDA complete.")
