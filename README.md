# ?? BMI Estimator — ML Minor Project

**Author:** Vishwakarma Aryan Santosh  
**Subject:** Machine Learning Minor Project  
**Year:** 2025–26

---

## ?? Problem Statement
Estimate Body Mass Index (BMI) using **Gender**, **Height**, and **Weight** as input features, and classify the result into standard BMI categories using Machine Learning.

## ?? Objectives
- Build a regression model to **predict the BMI value**
- Build a classification model to **predict the BMI category**
- Deploy an interactive web application using Streamlit

## ?? BMI Formula
```
BMI = Weight (kg) / Height (m)²
```

## ??? BMI Categories
| BMI Range     | Category       |
|---------------|----------------|
| < 18.5        | Underweight    |
| 18.5 – 24.9   | Normal weight  |
| 25.0 – 29.9   | Overweight     |
| = 30.0        | Obese          |

## ?? ML Models Used

### Regression (BMI Value Estimation)
- Linear Regression
- Random Forest Regressor ? *(Best)*
- Gradient Boosting Regressor

### Classification (BMI Category)
- Logistic Regression
- Decision Tree
- Random Forest Classifier ? *(Best)*
- K-Nearest Neighbors

## ?? Project Structure
```
bmi-estimator/
¦
+-- generate_dataset.py   # Generate synthetic dataset
+-- train_model.py        # Train & evaluate all models + save plots
+-- app.py                # Streamlit web application
+-- requirements.txt      # Python dependencies
+-- data/
¦   +-- bmi_dataset.csv   # Generated dataset (5000 samples)
+-- models/
¦   +-- bmi_regressor.pkl
¦   +-- bmi_classifier.pkl
¦   +-- scaler.pkl
¦   +-- label_encoder.pkl
¦   +-- cat_encoder.pkl
+-- plots/
    +-- 01_bmi_distribution.png
    +-- 02_scatter_height_weight.png
    +-- 03_bmi_by_gender.png
    +-- 04_regression_comparison.png
    +-- 05_classifier_comparison.png
    +-- 06_confusion_matrix.png
    +-- 07_actual_vs_predicted.png
```

## ?? How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Generate Dataset
```bash
python generate_dataset.py
```

### Step 3 — Train Models
```bash
python train_model.py
```

### Step 4 — Launch Web App
```bash
streamlit run app.py
```

## ?? Features (Input)
| Feature    | Type        | Description             |
|------------|-------------|-------------------------|
| Gender     | Categorical | Male / Female           |
| Height     | Numerical   | Height in cm (or ft/in) |
| Weight     | Numerical   | Weight in kg (or lbs)   |

## ?? Output
| Output         | Type        | Description                  |
|----------------|-------------|------------------------------|
| BMI Value      | Continuous  | Estimated BMI (e.g., 23.4)   |
| BMI Category   | Categorical | Underweight / Normal / Obese |
