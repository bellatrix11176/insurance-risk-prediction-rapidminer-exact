from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

# ============================================================
# InsuranceDecisionTree_MenuTree.py
# Builds a RapidMiner-style text tree and SAVES it to a .txt file
# ============================================================

BASE = Path(r"C:\Users\gigih\OneDrive\School\Week4")
FILE_XLSX = BASE / "InsuranceData.xlsx"
SCORED_CSV = BASE / "InsuranceData_NewApplicants_scored.csv"

TRAIN_SHEET = "Issued Policies"
LABEL = "Insurance Category"  # must match Excel exactly
RANDOM_STATE = 42

OUT_TREE_TXT = BASE / "InsuranceDecisionTree_MenuTree.txt"  # <--- saved here


def normalize_yes_no(series: pd.Series) -> pd.Series:
    """Normalize messy Yes/No-ish strings to 'Yes'/'No'."""
    s = series.astype(str).str.strip().str.lower()
    s = s.replace({
        "y": "yes", "yes": "yes", "true": "yes", "1": "yes",
        "n": "no",  "no": "no",  "false": "no", "0": "no",
        "nan": np.nan
    })
    return s.str.title()


def main():
    # -----------------------------
    # Load training from Excel
    # -----------------------------
    train = pd.read_excel(FILE_XLSX, sheet_name=TRAIN_SHEET).dropna(how="all")

    if LABEL not in train.columns:
        raise ValueError(
            f"Label column '{LABEL}' not found.\n"
            f"Columns found: {list(train.columns)}\n"
            f"Fix LABEL to match exactly (including spaces)."
        )

    # -----------------------------
    # Basic cleanup
    # -----------------------------
    train = train.copy()

    for col in ["Moving Violations", "Comprehensive Claims", "Late Payments"]:
        if col in train.columns:
            train[col] = normalize_yes_no(train[col])

    y = train[LABEL].astype(str).str.strip()
    X = train.drop(columns=[LABEL])

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop"
    )

    tree = DecisionTreeClassifier(random_state=RANDOM_STATE)

    model = Pipeline(steps=[
        ("prep", preprocessor),
        ("tree", tree)
    ])

    model.fit(X, y)

    # -----------------------------
    # Build feature names for export_text
    # -----------------------------
    prep = model.named_steps["prep"]
    feature_names = []
    feature_names.extend(numeric_cols)

    if categorical_cols:
        ohe = prep.named_transformers_["cat"]
        ohe_names = ohe.get_feature_names_out(categorical_cols).tolist()
        feature_names.extend(ohe_names)

    tree_estimator = model.named_steps["tree"]

    # -----------------------------
    # Export tree to string
    # -----------------------------
    text_tree = export_text(
        tree_estimator,
        feature_names=feature_names,
        decimals=3
    )

    header = (
        "============================================================\n"
        "DECISION TREE (Python export_text)\n"
        "============================================================\n\n"
    )

    output_text = header + text_tree

    # -----------------------------
    # SAVE TO TEXT FILE
    # -----------------------------
    OUT_TREE_TXT.write_text(output_text, encoding="utf-8")

    # Also print where it saved + show it
    print(f"\nSaved menu tree to:\n  {OUT_TREE_TXT}\n")
    print(output_text)

    # -----------------------------
    # Optional: quick scored CSV summary
    # -----------------------------
    if SCORED_CSV.exists():
        scored = pd.read_csv(SCORED_CSV)
        pred_col_candidates = [c for c in scored.columns if c.lower().startswith("prediction")]
        if pred_col_candidates:
            pred_col = pred_col_candidates[0]
            print("\n============================================================")
            print("SCORED CSV SUMMARY")
            print("============================================================")
            print("Prediction column:", pred_col)
            print(scored[pred_col].value_counts(dropna=False))
        else:
            print("\n(Note) Scored CSV loaded, but no prediction(*) column found.")
    else:
        print(f"\n(Note) Scored CSV not found at: {SCORED_CSV}")


if __name__ == "__main__":
    main()
