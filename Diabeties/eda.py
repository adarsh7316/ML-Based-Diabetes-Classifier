# eda.py
# A simple script to do Exploratory Data Analysis (EDA) on the diabetes dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned data
print("Loading cleaned data...")
df = pd.read_csv("diabetes_cleaned.csv")

print("Starting to make plots. Please wait...")

# Target distribution (How many have diabetes?)
print("Making Plot 1: Diabetes Count")
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=df, x='Diabetes_binary', hue='Diabetes_binary', palette='Set1', legend=False)
for container in ax.containers:
    ax.bar_label(container)
plt.title("Count of Diabetes Cases (0 = No, 1 = Yes)")
plt.savefig("eda_outputs/01_diabetes_count.png")
plt.close()

# BMI Histogram
print("Making Plot 2: BMI Histogram")
plt.figure(figsize=(8, 5))
ax = sns.histplot(data=df, x='BMI', bins=20, color='skyblue', edgecolor='black')
for container in ax.containers:
    ax.bar_label(container, fontsize=8, padding=2)
plt.title("Distribution of BMI")
plt.xlabel("BMI (Body Mass Index)")
plt.ylabel("Frequency")
plt.savefig("eda_outputs/02_bmi_hist.png")
plt.close()

# High Blood Pressure vs Diabetes
print("Making Plot 3: High Blood Pressure vs Diabetes")
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=df, x='HighBP', hue='HighBP', palette='Set1', legend=False)
for container in ax.containers:
    ax.bar_label(container)
plt.title("HighBP and Diabetes")
plt.savefig("eda_outputs/03_highbp.png")
plt.close()

# High Cholesterol vs Diabetes
print("Making Plot 4: High Cholesterol vs Diabetes")
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=df, x='HighChol', hue='HighChol', palette='Set1', legend=False)
for container in ax.containers:
    ax.bar_label(container)
plt.title("High Cholesterol and Diabetes")
plt.savefig("eda_outputs/04_highchol.png")
plt.close()

# Smoker vs Diabetes
print("Making Plot 5: Smoker vs Diabetes")
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=df, x='Smoker', hue='Smoker', palette='Set1', legend=False)
for container in ax.containers:
    ax.bar_label(container)
plt.title("Smoking Status and Diabetes")
plt.savefig("eda_outputs/05_smoker.png")
plt.close()

# Stroke vs Diabetes
print("Making Plot 6: Stroke vs Diabetes")
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=df, x='Stroke', hue='Stroke', palette='Set1', legend=False)
for container in ax.containers:
    ax.bar_label(container)
plt.title("Stroke History and Diabetes")
plt.savefig("eda_outputs/06_stroke.png")
plt.close()

# Heart Disease vs Diabetes
print("Making Plot 7: Heart Disease vs Diabetes")
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=df, x='HeartDiseaseorAttack', hue='HeartDiseaseorAttack', palette='Set1', legend=False)
for container in ax.containers:
    ax.bar_label(container)
plt.title("Heart Disease and Diabetes")
plt.savefig("eda_outputs/07_heart_disease.png")
plt.close()

# Physical Activity vs Diabetes
print("Making Plot 8: Physical Activity vs Diabetes")
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=df, x='PhysActivity', hue='PhysActivity', palette='Set1', legend=False)
for container in ax.containers:
    ax.bar_label(container)
plt.title("Physical Activity and Diabetes")
plt.savefig("eda_outputs/08_activity.png")
plt.close()

# Eating Fruits vs Diabetes
print("Making Plot 9: Eating Fruits vs Diabetes")
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=df, x='Fruits', hue='Fruits', palette='Set1', legend=False)
for container in ax.containers:
    ax.bar_label(container)
plt.title("Eating Fruits and Diabetes")
plt.savefig("eda_outputs/09_fruits.png")
plt.close()

# Age Group and Diabetes
print("Making Plot 10: Age Group and Diabetes")
plt.figure(figsize=(10, 5))
ax = sns.countplot(data=df, x='Age', hue='Diabetes_binary', palette='Set1')
for container in ax.containers:
    ax.bar_label(container, fontsize=8)
plt.title("Age Groups and Diabetes")
plt.xlabel("Age Category (1 = 18-24, 13 = 80+)")
plt.savefig("eda_outputs/10_age.png")
plt.close()

# General Health vs Diabetes
print("Making Plot 11: General Health vs Diabetes")
plt.figure(figsize=(10, 5))
ax = sns.countplot(data=df, x='GenHlth', hue='Diabetes_binary', palette='Set1')
for container in ax.containers:
    ax.bar_label(container, fontsize=8)
plt.title("General Health and Diabetes")
plt.xlabel("General Health (1 = Excellent, 5 = Poor)")
plt.savefig("eda_outputs/11_health.png")
plt.close()

# Difficulty Walking vs Diabetes
print("Making Plot 12: Difficulty Walking vs Diabetes")
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=df, x='DiffWalk', hue='Diabetes_binary', palette='Set1')
for container in ax.containers:
    ax.bar_label(container, fontsize=8)
plt.title("Difficulty Walking and Diabetes")
plt.savefig("eda_outputs/12_diffwalk.png")
plt.close()

# Correlation Heatmap
print("Making Plot 13: Correlation Matrix Heatmap")
plt.figure(figsize=(14, 12))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 8})
plt.title("Correlation between all variables")
plt.savefig("eda_outputs/13_heatmap.png")
plt.close()


print("\nMoving on to the Deep Dive Plots...\n")

# Gender Gap Comparison
print("Making Plot 14: Gender Gap")
plt.figure(figsize=(6, 5))
ax = sns.barplot(data=df, x='Sex', y='Diabetes_binary', hue='Sex', palette='Set2', legend=False)
for container in ax.containers:
    
    ax.bar_label(container, fmt='%.3f', padding=3)
plt.title("Diabetes Percentage by Gender (0 = Female, 1 = Male)")
plt.ylabel("Likelihood of Diabetes (%)")
plt.savefig("eda_outputs/14_gender_gap.png")
plt.close()

# The Paradox Group (Weight vs Good Vitals)
print("Making Plot 15: Weight vs Health Paradox")
def categorize_health(row):
    bmi = row['BMI']
    bad_bp = row['HighBP']
    bad_chol = row['HighChol']
    
    if bmi < 25 and bad_bp == 0 and bad_chol == 0:
        return "Normal Weight,\nHealthy Vitals"
    elif bmi >= 30 and bad_bp == 0 and bad_chol == 0:
        return "Obese,\nHealthy Vitals"
    elif bmi < 25 and bad_bp == 1 and bad_chol == 1:
        return "Normal Weight,\nBad Vitals"
    elif bmi >= 30 and bad_bp == 1 and bad_chol == 1:
        return "Obese,\nBad Vitals"
    else:
        return "Other"

df['Paradox_Group'] = df.apply(categorize_health, axis=1)
paradox_data = df[df['Paradox_Group'] != "Other"]

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=paradox_data, x='Paradox_Group', y='Diabetes_binary', hue='Paradox_Group', palette='Set2', legend=False)
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', padding=3)
plt.title("The Paradox: Weight vs Blood/Cholesterol Vitals")
plt.ylabel("Likelihood of Diabetes (%)")
plt.savefig("eda_outputs/15_paradox.png")
plt.close()

# Snowball of Bad Habits
print("Making Plot 16: Bad Habits Snowball")
def count_bad_habits(row):
    habits = 0
    if row['Smoker'] == 1: 
        habits += 1
    if row['PhysActivity'] == 0:  
        habits += 1 
    if row['HvyAlcoholConsump'] == 1: 
        habits += 1
    return habits

df['Bad_Habits_Count'] = df.apply(count_bad_habits, axis=1)

plt.figure(figsize=(6, 5))
ax = sns.barplot(data=df, x='Bad_Habits_Count', y='Diabetes_binary', hue='Bad_Habits_Count', palette='Set2', legend=False)
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', padding=3)
plt.title("Number of Bad Habits vs Diabetes")
plt.xlabel("Bad Habits Count (Smoking, Inactivity, Heavy Drinking)")
plt.ylabel("Likelihood of Diabetes (%)")
plt.savefig("eda_outputs/16_bad_habits.png")
plt.close()

# Age vs Health Matrix
print("Making Plot 17: Age vs Health Danger Zone Matrix")
plt.figure(figsize=(10, 6))
matrix = pd.pivot_table(df, values='Diabetes_binary', index='GenHlth', columns='Age')

sns.heatmap(matrix, annot=True, cmap='YlOrRd', fmt='.2f', annot_kws={"size": 8})
plt.title("Age vs General Health 'Danger Zone'")
plt.xlabel("Age Group (Older to the right ->)")
plt.ylabel("General Health (1=Excellent, 5=Poor)")
plt.savefig("eda_outputs/17_health_matrix.png")
plt.close()

# BP and Cholesterol Trifecta
print("Making Plot 18: BP and Cholesterol Combo")
def get_combo(row):
    bp = row['HighBP']
    chol = row['HighChol']
    if bp == 1 and chol == 1:
        return "Both Bad"
    elif bp == 1:
        return "Only BP"
    elif chol == 1:
        return "Only Chol"
    else:
        return "Neither"

df['BP_Chol_Combo'] = df.apply(get_combo, axis=1)

plt.figure(figsize=(8, 5))
ax = sns.barplot(data=df, x='BP_Chol_Combo', y='Diabetes_binary', hue='BP_Chol_Combo', palette='Set2', legend=False)
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', padding=3)
plt.title("High BP & Cholesterol vs Diabetes")
plt.ylabel("Likelihood of Diabetes (%)")
plt.savefig("eda_outputs/18_combo.png")
plt.close()

# The Couch Potato
print("Making Plot 19: The Couch Potato Penalty")
plt.figure(figsize=(8, 5))
ax = sns.barplot(data=df, x='PhysActivity', y='Diabetes_binary', hue='PhysActivity', palette='Set2', legend=False)
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', padding=3)
plt.title("Physical Activity: Inactive (0) vs Active (1)")
plt.ylabel("Likelihood of Diabetes (%)")
plt.savefig("eda_outputs/19_couch_potato.png")
plt.close()

# Mentally vs Physically Bad Days
print("Making Plot 20: Mentally vs Physically Bad Days")
def check_bad_days(row):
    phys = row['PhysHlth']
    ment = row['MentHlth']
    if phys > 14 and ment > 14:
        return "Both Bad"
    elif phys > 14:
        return "Physical Bad"
    elif ment > 14:
        return "Mental Bad"
    else:
        return "Mostly Good Days"

df['Bad_Days_Status'] = df.apply(check_bad_days, axis=1)

plt.figure(figsize=(8, 5))
ax = sns.barplot(data=df, x='Bad_Days_Status', y='Diabetes_binary', hue='Bad_Days_Status', palette='Set2', legend=False)
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', padding=3)
plt.title("Experiencing 14+ Bad Days vs Diabetes")
plt.ylabel("Likelihood of Diabetes (%)")
plt.savefig("eda_outputs/20_bad_days.png")
plt.close()

print("\nAwesome! All 20 plots are successfully generated with data labels on top of bars!")
print("You can open the folder 'eda_outputs' to check them.")
