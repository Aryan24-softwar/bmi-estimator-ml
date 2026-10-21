"""
Generate a synthetic BMI dataset for training and testing.
Dataset: Gender, Height (cm), Weight (kg) -> BMI, BMI Category
Author : Vishwakarma Aryan Santosh
Project: ML Minor Project - BMI Estimation
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 5000


def generate_bmi_dataset(n=N):
    genders = np.random.choice(["Male", "Female"], size=n)
    heights = []
    weights = []

    for g in genders:
        if g == "Male":
            h = np.random.normal(loc=175, scale=7)
            w = np.random.normal(loc=78,  scale=12)
        else:
            h = np.random.normal(loc=162, scale=6)
            w = np.random.normal(loc=64,  scale=10)
        h = float(np.clip(h, 140, 210))
        w = float(np.clip(w, 35,  150))
        heights.append(round(h, 1))
        weights.append(round(w, 1))

    heights = np.array(heights)
    weights = np.array(weights)
    bmi = np.round(weights / ((heights / 100) ** 2), 2)

    def categorise(b):
        if b < 18.5:  return "Underweight"
        elif b < 25.0: return "Normal weight"
        elif b < 30.0: return "Overweight"
        else:          return "Obese"

    df = pd.DataFrame({
        "Gender":       genders,
        "Height_cm":    heights,
        "Weight_kg":    weights,
        "BMI":          bmi,
        "BMI_Category": [categorise(b) for b in bmi],
    })
    return df


if __name__ == "__main__":
    df = generate_bmi_dataset()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bmi_dataset.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Dataset saved -> {out}")
    print(df.head())
    print("\nCategory distribution:\n", df["BMI_Category"].value_counts())
