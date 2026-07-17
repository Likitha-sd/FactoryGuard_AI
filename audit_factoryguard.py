import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

print("=" * 80)
print("FACTORYGUARD AI - PROJECT AUDIT")
print("=" * 80)

# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

DATA_PATH = "data/processed/factory_sensor_features.csv"

df = pd.read_csv(DATA_PATH)

print("\nDATASET")
print("-" * 80)

print("Shape:", df.shape)
print("Rows :", len(df))
print("Columns :", len(df.columns))

print("\nColumns")
for c in df.columns:
    print(" -", c)

print("\nMachines")
print(df["machine_id"].value_counts())

print("\nMachine Failure Distribution")
print(df["machine_failure"].value_counts())

# ------------------------------------------------------------------
# Features
# ------------------------------------------------------------------

TARGET = "machine_failure"

DROP_COLUMNS = ["timestamp", TARGET]

feature_columns = [c for c in df.columns if c not in DROP_COLUMNS]

print("\nFEATURES")
print("-" * 80)

print("Raw + Engineered Features :", len(feature_columns))

for f in feature_columns:
    print(f)

# ------------------------------------------------------------------
# Load Model
# ------------------------------------------------------------------

MODEL_PATH = "models/random_forest.pkl"

print("\nMODEL")
print("-" * 80)

model = joblib.load(MODEL_PATH)

print(model)

if hasattr(model, "n_estimators"):
    print("Trees :", model.n_estimators)

if hasattr(model, "max_depth"):
    print("Max Depth :", model.max_depth)

if hasattr(model, "random_state"):
    print("Random State :", model.random_state)

# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

TEST_PATH = "data/processed/test.csv"

if os.path.exists(TEST_PATH):

    print("\nMODEL EVALUATION")
    print("-" * 80)

    test_df = pd.read_csv(TEST_PATH)

    X = test_df.drop(columns=["timestamp", TARGET])

    y = test_df[TARGET]

    preds = model.predict(X)

    probs = model.predict_proba(X)[:, 1]

    print("Accuracy :", accuracy_score(y, preds))
    print("Precision :", precision_score(y, preds))
    print("Recall :", recall_score(y, preds))
    print("F1 :", f1_score(y, preds))
    print("ROC-AUC :", roc_auc_score(y, probs))

    print("\nConfusion Matrix")

    print(confusion_matrix(y, preds))

    print("\nClassification Report")

    print(classification_report(y, preds))

else:

    print("\nNo saved test.csv found.")
    print("Training metrics cannot be reproduced.")

# ------------------------------------------------------------------
# Feature Importance
# ------------------------------------------------------------------

if hasattr(model, "feature_importances_"):

    print("\nTOP 15 FEATURES")
    print("-" * 80)

    importance = pd.DataFrame(
        {
            "Feature": feature_columns,
            "Importance": model.feature_importances_,
        }
    )

    importance = importance.sort_values(
        "Importance",
        ascending=False,
    )

    print(importance.head(15))

print("\nEND OF AUDIT")
print("=" * 80)