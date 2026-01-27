from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text


# ============================================================
# FILE PATHS
# ============================================================
BASE = Path(r"C:\Users\gigih\OneDrive\School\Week4")
EXCEL_FILE = BASE / "InsuranceData.xlsx"

TRAIN_SHEET = "Issued Policies"
SCORE_SHEET = "New Applicants"

LABEL_COL = "Insurance Category"

TREE_TXT = BASE / "InsuranceDecisionTree_MenuTree.txt"
SCORED_CSV = BASE / "InsuranceData_NewApplicants_scored_by_python.csv"


# ============================================================
# UTILITIES
# ============================================================
def clean_columns(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def prettify_tree(tree_text):
    """
    Converts one-hot splits like:
      Payment Method_Credit Card <= 0.5
    into:
      Payment Method != Credit Card
    """
    out = []
    for line in tree_text.splitlines():
        indent = line[:len(line) - len(line.lstrip())]
        content = line.strip()

        if ("<=" in content or ">" in content) and "_" in content:
            parts = content.split()
            for i, p in enumerate(parts):
                if p in ("<=", ">"):
                    feature = " ".join(parts[:i])
                    op = parts[i]
                    try:
                        thresh = float(parts[i + 1])
                    except:
                        continue

                    if abs(thresh - 0.5) < 1e-6:
                        base, category = feature.split("_", 1)
                        if op == "<=":
                            out.append(f"{indent}{base} != {category}")
                            break
                        else:
                            out.append(f"{indent}{base} = {category}")
                            break
            else:
                out.append(line)
        else:
            out.append(line)

    return "\n".join(out)


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n================ LOADING DATA =================")

    train = pd.read_excel(EXCEL_FILE, sheet_name=TRAIN_SHEET).dropna(how="all")
    score = pd.read_excel(EXCEL_FILE, sheet_name=SCORE_SHEET).dropna(how="all")

    train = clean_columns(train)
    score = clean_columns(score)

    print("Training shape:", train.shape)
    print("Scoring shape :", score.shape)

    if LABEL_COL not in train.columns:
        raise ValueError(f"Label column not found: {LABEL_COL}")

    y = train[LABEL_COL].astype(str)
    X = train.drop(columns=[LABEL_COL])

    missing = set(X.columns) - set(score.columns)
    if missing:
        raise ValueError(f"Scoring data missing columns: {missing}")

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    print("\nNumeric columns:", numeric_cols)
    print("Categorical columns:", categorical_cols)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )

    tree = DecisionTreeClassifier(random_state=42)

    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("tree", tree)
        ]
    )

    print("\n================ TRAINING TREE =================")
    pipeline.fit(X, y)

    encoder = pipeline.named_steps["prep"].named_transformers_["cat"]
    encoded_cat_features = encoder.get_feature_names_out(categorical_cols)

    feature_names = numeric_cols + encoded_cat_features.tolist()

    raw_tree = export_text(
        pipeline.named_steps["tree"],
        feature_names=feature_names,
        decimals=3
    )

    pretty_tree = prettify_tree(raw_tree)

    TREE_TXT.write_text(pretty_tree, encoding="utf-8")

    print("\nTree written to:")
    print(TREE_TXT)

    print("\n================ SCORING NEW APPLICANTS =================")

    probabilities = pipeline.predict_proba(score[X.columns])
    predictions = pipeline.predict(score[X.columns])
    classes = pipeline.classes_

    scored = score.copy()
    scored[f"prediction({LABEL_COL})"] = predictions

    for i, cls in enumerate(classes):
        scored[f"confidence({cls})"] = probabilities[:, i]

    scored.to_csv(SCORED_CSV, index=False)

    print("Scored data saved to:")
    print(SCORED_CSV)

    print("\n================ PREVIEW =================")
    cols = [f"prediction({LABEL_COL})"] + [f"confidence({c})" for c in classes]
    print(scored[cols].head(8))

    print("\n✅ DONE. Open the .txt file to view the full decision tree.")


if __name__ == "__main__":
    main()
