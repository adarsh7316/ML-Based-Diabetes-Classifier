# clean.py
# A simple script to clean the diabetes dataset

import pandas as pd

# Step 1: Load the data
print("Step 1: Loading data...")
df = pd.read_csv("final.csv")
print("Data loaded! Rows:", len(df), "Columns:", len(df.columns))

# Show the first few rows
print("Here is how the data looks:")
print(df.head())

print(df.describe())

# Step 2: Check for missing values
print("\nStep 2: Checking for missing values...")
missing_count = df.isnull().sum().sum()
if missing_count > 0:
    print("Found missing values! Dropping them now.")
    df = df.dropna()
    print("New size of data:", len(df))
else:
    print("No missing values found.")

# Step 3: Remove duplicate rows
print("\nStep 3: Removing any duplicate rows...")
original_length = len(df)
df = df.drop_duplicates()
duplicates_removed = original_length - len(df)
print("Duplicates removed:", duplicates_removed)
print("Rows left:", len(df))

# Step 4: Clean the BMI column
# We only want BMI values between 15 and 60
print("\nStep 4: Cleaning BMI values...")
original_length = len(df)

# Keep BMI greater than or equal to 15
df = df[df['BMI'] >= 15]

# Keep BMI less than or equal to 60
df = df[df['BMI'] <= 60]

outliers_removed = original_length - len(df)
print("BMI Outliers removed:", outliers_removed)
print("Rows left:", len(df))

print(df.describe())
# Step 5: Save the clean dataset
print("\nStep 5: Saving the clean data to a new file...")
df.to_csv("diabetes_cleaned.csv", index=False)
print("Data cleaning is done! The file 'diabetes_cleaned.csv' is ready.")