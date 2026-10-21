# coding: utf-8
"""
Train ML models to estimate BMI and predict BMI Category.
Author : Vishwakarma Aryan Santosh
Project: ML Minor Project - BMI Estimation
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import LabelEncoder, StandardScaler
from sklearn.linear_model   import LinearRegression, LogisticRegression
from sklearn.ensemble       import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.tree           import DecisionTreeClassifier
from sklearn.neighbors      import KNeighborsClassifier
from sklearn.metrics        import (mean_absolute_error, mean_squared_error, r2_score,
                                    accuracy_score, classification_report, confusion_matrix)

warnings.filterwarnings("ignore")

BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "data",   "bmi_dataset.csv")
MODEL  = os.path.join(BASE, "models")
PLOTS  = os.path.join(BASE, "plots")
os.makedirs(MODEL, exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

print("=" * 60)
print("   ML MINOR PROJECT - BMI ESTIMATION")
print("   Author : Vishwakarma Aryan Santosh")
print("=" * 60)

df = pd.read_csv(DATA)
print(f"\nDataset shape: {df.shape}")
print(df.head())
print("\nBasic Stats:\n", df.describe())
print("\nMissing values:\n", df.isnull().sum())

le = LabelEncoder()
df["Gender_enc"] = le.fit_transform(df["Gender"])

FEATURES = ["Gender_enc", "Height_cm", "Weight_kg"]
X = df[FEATURES]
y_bmi      = df["BMI"]
y_category = df["BMI_Category"]

cat_enc = LabelEncoder()
y_cat_enc = cat_enc.fit_transform(y_category)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tr, X_te, y_bmi_tr, y_bmi_te = train_test_split(X_scaled, y_bmi,     test_size=0.2, random_state=42)
_,    _,    y_cat_tr, y_cat_te  = train_test_split(X_scaled, y_cat_enc, test_size=0.2, random_state=42)

print("\n" + "-"*60)
print("TASK 1: BMI VALUE ESTIMATION (Regression)")
print("-"*60)

regressors = {
    "Linear Regression":           LinearRegression(),
    "Random Forest Regressor":     RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
}

best_reg_name, best_reg_model, best_r2 = "", None, -999
reg_results = []

for name, model in regressors.items():
    model.fit(X_tr, y_bmi_tr)
    preds = model.predict(X_te)
    mae  = mean_absolute_error(y_bmi_te, preds)
    rmse = np.sqrt(mean_squared_error(y_bmi_te, preds))
    r2   = r2_score(y_bmi_te, preds)
    reg_results.append({"Model": name, "MAE": round(mae,4), "RMSE": round(rmse,4), "R2": round(r2,4)})
    print(f"\n{name}\n  MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")
    if r2 > best_r2:
        best_r2, best_reg_name, best_reg_model = r2, name, model

print(f"\nBest Regressor : {best_reg_name}  (R2={best_r2:.4f})")

print("\n" + "-"*60)
print("TASK 2: BMI CATEGORY PREDICTION (Classification)")
print("-"*60)

classifiers = {
    "Logistic Regression":   LogisticRegression(max_iter=500, random_state=42),
    "Decision Tree":         DecisionTreeClassifier(random_state=42),
    "Random Forest":         RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN":                   KNeighborsClassifier(n_neighbors=5),
}

best_clf_name, best_clf_model, best_acc = "", None, 0
clf_results = []

for name, model in classifiers.items():
    model.fit(X_tr, y_cat_tr)
    preds = model.predict(X_te)
    acc   = accuracy_score(y_cat_te, preds)
    clf_results.append({"Model": name, "Accuracy": round(acc*100, 2)})
    print(f"\n{name}\n  Accuracy = {acc*100:.2f}%")
    if acc > best_acc:
        best_acc, best_clf_name, best_clf_model = acc, name, model

print(f"\nBest Classifier : {best_clf_name}  (Accuracy={best_acc*100:.2f}%)")
print("\nDetailed Report:")
print(classification_report(y_cat_te, best_clf_model.predict(X_te), target_names=cat_enc.classes_))

pickle.dump(best_reg_model,  open(os.path.join(MODEL, "bmi_regressor.pkl"),  "wb"))
pickle.dump(best_clf_model,  open(os.path.join(MODEL, "bmi_classifier.pkl"), "wb"))
pickle.dump(scaler,          open(os.path.join(MODEL, "scaler.pkl"),         "wb"))
pickle.dump(le,              open(os.path.join(MODEL, "label_encoder.pkl"),  "wb"))
pickle.dump(cat_enc,         open(os.path.join(MODEL, "cat_encoder.pkl"),    "wb"))
print("\nAll models saved to /models/")

plt.style.use("seaborn-v0_8-whitegrid")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df["BMI"], kde=True, color="#4C72B0", ax=axes[0])
axes[0].set_title("BMI Distribution", fontsize=14, fontweight="bold")
axes[0].set_xlabel("BMI")
order = ["Underweight", "Normal weight", "Overweight", "Obese"]
colors = ["#2196F3","#4CAF50","#FF9800","#F44336"]
cat_counts = df["BMI_Category"].value_counts().reindex(order)
axes[1].bar(order, cat_counts.values, color=colors)
axes[1].set_title("BMI Category Distribution", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "01_bmi_distribution.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
palette = {"Underweight":"#2196F3","Normal weight":"#4CAF50","Overweight":"#FF9800","Obese":"#F44336"}
for cat, grp in df.groupby("BMI_Category"):
    ax.scatter(grp["Height_cm"], grp["Weight_kg"], label=cat, alpha=0.5, s=15, color=palette.get(cat,"gray"))
ax.set_xlabel("Height (cm)", fontsize=12)
ax.set_ylabel("Weight (kg)", fontsize=12)
ax.set_title("Height vs Weight by BMI Category", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "02_scatter_height_weight.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="Gender", y="BMI", palette=["#42A5F5","#EC407A"], ax=ax)
ax.set_title("BMI by Gender", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "03_bmi_by_gender.png"), dpi=150)
plt.close()

reg_df = pd.DataFrame(reg_results)
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(reg_df["Model"], reg_df["R2"], color=["#5C6BC0","#26A69A","#FFA726"])
ax.set_ylim(0, 1.05)
ax.set_ylabel("R2 Score", fontsize=12)
ax.set_title("Regression Model Comparison (R2)", fontsize=14, fontweight="bold")
for bar, val in zip(bars, reg_df["R2"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "04_regression_comparison.png"), dpi=150)
plt.close()

clf_df = pd.DataFrame(clf_results)
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(clf_df["Model"], clf_df["Accuracy"], color=["#EF5350","#AB47BC","#42A5F5","#26A69A"])
ax.set_ylim(0, 110)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Classifier Accuracy Comparison", fontsize=14, fontweight="bold")
for bar, val in zip(bars, clf_df["Accuracy"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "05_classifier_comparison.png"), dpi=150)
plt.close()

cm = confusion_matrix(y_cat_te, best_clf_model.predict(X_te))
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=cat_enc.classes_, yticklabels=cat_enc.classes_, ax=ax)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("Actual",    fontsize=12)
ax.set_title(f"Confusion Matrix - {best_clf_name}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "06_confusion_matrix.png"), dpi=150)
plt.close()

preds_best = best_reg_model.predict(X_te)
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_bmi_te, preds_best, alpha=0.4, s=15, color="#5C6BC0")
mn, mx = y_bmi_te.min(), y_bmi_te.max()
ax.plot([mn, mx], [mn, mx], "r--", lw=2, label="Ideal")
ax.set_xlabel("Actual BMI",    fontsize=12)
ax.set_ylabel("Predicted BMI", fontsize=12)
ax.set_title(f"Actual vs Predicted BMI - {best_reg_name}", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "07_actual_vs_predicted.png"), dpi=150)
plt.close()

print("\nAll plots saved to /plots/")
print("\nTraining complete!")