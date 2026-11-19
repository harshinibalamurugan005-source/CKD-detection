# Chronic Kidney Disease (CKD) Detection

A data science and machine learning mini project that predicts whether a patient has chronic kidney disease from blood, urine, and medical history features.

**Type:** Supervised ML, binary classification (CKD / Not CKD)

## Project Workflow
1. Data loading and inspection
2. Data cleaning (hidden whitespace and tabs in labels, `?` placeholders, numbers stored as text, duplicates, missing values)
3. Exploratory data analysis (class balance, distributions, boxplots, correlation heatmap)
4. Train/test split (80/20, stratified)
5. Preprocessing pipeline (median/mode imputation, scaling, one-hot encoding), fitted on training data only to avoid data leakage
6. Model comparison with 5-fold cross-validation: Logistic Regression, Decision Tree, Random Forest
7. Hyperparameter tuning with GridSearchCV
8. Evaluation on the unseen test set
9. Feature importance and a new-patient prediction

## Results (Random Forest, test set)
| Metric | Score |
|---|---|
| Accuracy | 98.8% |
| Precision | 100% |
| Recall | 98% |
| F1-score | 99% |
| ROC-AUC | 1.00 |

## Files
- `CKD_Detection.ipynb` - the full notebook
- `kidney_disease.csv` - dataset
- `kidney_disease_clean.csv` - cleaned dataset
- `ckd_model.joblib` - trained model

## How to Run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
jupyter notebook CKD_Detection.ipynb
```

## Notes
The dataset has the same structure as the UCI/Kaggle CKD dataset (about 400 rows, 24 features). The dataset is small and the classes are very separable, so the high accuracy may not hold on larger real-world data. This project is for learning only and is not a medical diagnostic tool.

## Tech Stack
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn
