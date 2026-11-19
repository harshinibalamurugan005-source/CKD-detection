import warnings; warnings.filterwarnings("ignore")
import os
os.environ["MPLCONFIGDIR"] = r"C:\Users\ELCOT\AppData\Local\Temp\mplcfg"

import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay,
                             classification_report)

sns.set_theme(style="whitegrid")
RANDOM_STATE = 42

# ── 1. Load ──────────────────────────────────────────────────────────────────
df = pd.read_csv("kidney_disease.csv")
print("=" * 55)
print("SECTION 1 — DATA SHAPE & RAW INFO")
print("=" * 55)
print("Shape:", df.shape)
print("Target values:", df["classification"].unique())
print("dm values    :", df["dm"].unique())

# ── 2. Clean ─────────────────────────────────────────────────────────────────
clean = df.copy()
clean = clean.drop(columns=["id"], errors="ignore")
text_cols = clean.select_dtypes(include=["object", "string"]).columns
for c in text_cols:
    clean[c] = clean[c].astype(str).str.strip().replace({"?": np.nan, "nan": np.nan, "": np.nan})
num_cols = ["age","bp","sg","al","su","bgr","bu","sc","sod","pot","hemo","pcv","wc","rc"]
for c in num_cols:
    clean[c] = pd.to_numeric(clean[c], errors="coerce")
clean["classification"] = clean["classification"].map({"ckd": 1, "notckd": 0})
clean = clean.drop_duplicates().dropna(subset=["classification"]).reset_index(drop=True)
clean["classification"] = clean["classification"].astype(int)

print("\n" + "=" * 55)
print("SECTION 2 — AFTER CLEANING")
print("=" * 55)
print("Shape after cleaning:", clean.shape)
print("\nTarget counts:\n", clean["classification"].value_counts().to_string())
print("\nMissing % per column (top 10):")
print((clean.isnull().mean() * 100).round(1).sort_values(ascending=False).head(10).to_string())
clean.to_csv("kidney_disease_clean.csv", index=False)

# ── 3. Split ──────────────────────────────────────────────────────────────────
cat_cols = [c for c in clean.columns if c not in num_cols + ["classification"]]
X = clean.drop(columns="classification")
y = clean["classification"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
print("\n" + "=" * 55)
print("SECTION 3 — TRAIN / TEST SPLIT")
print("=" * 55)
print("Train:", X_train.shape, "| Test:", X_test.shape)
print("CKD rate  train: %.3f  test: %.3f" % (y_train.mean(), y_test.mean()))

# ── 4. Pipeline ───────────────────────────────────────────────────────────────
num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                     ("onehot", OneHotEncoder(handle_unknown="ignore"))])
preprocess = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])

# ── 5. Cross-validation comparison ───────────────────────────────────────────
print("\n" + "=" * 55)
print("SECTION 4 — CROSS-VALIDATION (5-fold)")
print("=" * 55)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
}
cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
rows = []
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    acc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    f1  = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
    rows.append({"Model": name, "CV Accuracy": round(acc.mean(), 3),
                 "CV Acc Std": round(acc.std(), 3), "CV F1": round(f1.mean(), 3)})
cv_df = pd.DataFrame(rows).sort_values("CV F1", ascending=False)
print(cv_df.to_string(index=False))

# ── 6. GridSearchCV ───────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("SECTION 5 — HYPERPARAMETER TUNING (Random Forest)")
print("=" * 55)
rf_pipe = Pipeline([("prep", preprocess),
                    ("model", RandomForestClassifier(random_state=RANDOM_STATE))])
param_grid = {"model__n_estimators":    [100, 200],
              "model__max_depth":        [3, 5, None],
              "model__min_samples_leaf": [1, 2, 4]}
grid = GridSearchCV(rf_pipe, param_grid, cv=cv, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
print("Best CV F1 : %.3f" % grid.best_score_)

# ── 7. Final evaluation ───────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("SECTION 6 — FINAL TEST SET EVALUATION")
print("=" * 55)
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]
print("Accuracy :", round(accuracy_score(y_test, y_pred), 3))
print("Precision:", round(precision_score(y_test, y_pred), 3))
print("Recall   :", round(recall_score(y_test, y_pred), 3))
print("F1-score :", round(f1_score(y_test, y_pred), 3))
print("ROC-AUC  :", round(roc_auc_score(y_test, y_prob), 3))
print()
print(classification_report(y_test, y_pred, target_names=["Not CKD", "CKD"]))

# ── 8. Plots ──────────────────────────────────────────────────────────────────
# Class balance + missing
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
clean["classification"].value_counts().sort_index().plot.bar(ax=ax0, color=["#4C72B0","#C44E52"])
ax0.set_title("Class balance"); ax0.set_xticklabels(["Not CKD","CKD"], rotation=0)
(clean[num_cols].isnull().mean()*100).sort_values().plot.barh(ax=ax1, color="#55A868")
ax1.set_title("Missing % (numeric columns)"); plt.tight_layout()
plt.savefig("plot_class_missing.png", dpi=120); plt.close()

# Histograms
clean[num_cols].hist(bins=20, figsize=(14, 9), color="#4C72B0"); plt.tight_layout()
plt.savefig("plot_histograms.png", dpi=120); plt.close()

# Boxplots by class
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, col in zip(axes.ravel(), ["hemo","sc","bu","pcv","rc","sg","al","bgr"]):
    sns.boxplot(data=clean, x="classification", y=col, ax=ax, palette=["#4C72B0","#C44E52"])
    ax.set_xticklabels(["Not CKD","CKD"])
plt.tight_layout(); plt.savefig("plot_boxplots.png", dpi=120); plt.close()

# Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(clean[num_cols+["classification"]].corr(), annot=True, fmt=".1f",
            cmap="coolwarm", center=0)
plt.title("Correlation heatmap"); plt.tight_layout()
plt.savefig("plot_heatmap.png", dpi=120); plt.close()

# Confusion matrix + ROC
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred,
    display_labels=["Not CKD","CKD"], ax=ax0, cmap="Blues")
ax0.set_title("Confusion matrix")
RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax1)
ax1.plot([0,1],[0,1],"k--"); ax1.set_title("ROC curve")
plt.tight_layout(); plt.savefig("plot_confusion_roc.png", dpi=120); plt.close()

# Feature importance
names = best_model.named_steps["prep"].get_feature_names_out()
imp = pd.Series(best_model.named_steps["model"].feature_importances_, index=names)
imp.sort_values().tail(12).plot.barh(figsize=(8,5), color="#4C72B0")
plt.title("Top 12 features"); plt.tight_layout()
plt.savefig("plot_feature_importance.png", dpi=120); plt.close()

# ── 9. Save model + predict new patient ──────────────────────────────────────
joblib.dump(best_model, "ckd_model.joblib")
new_patient = pd.DataFrame([{
    "age":60,"bp":90,"sg":1.010,"al":3,"su":1,"rbc":"abnormal","pc":"abnormal",
    "pcc":"present","ba":"notpresent","bgr":180,"bu":85,"sc":4.2,"sod":130,"pot":5.0,
    "hemo":9.5,"pcv":28,"wc":9000,"rc":3.4,"htn":"yes","dm":"yes","cad":"no",
    "appet":"poor","pe":"yes","ane":"yes"}])
loaded = joblib.load("ckd_model.joblib")
prob = loaded.predict_proba(new_patient)[0, 1]
print("\n" + "=" * 55)
print("SECTION 7 — NEW PATIENT PREDICTION")
print("=" * 55)
print("Probability of CKD: %.1f%%  ->  %s" % (
    prob * 100, "CKD likely" if prob >= 0.5 else "Not CKD"))
print("\nAll plots saved successfully.")
