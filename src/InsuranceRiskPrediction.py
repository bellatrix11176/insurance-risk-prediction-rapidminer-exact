# InsuranceRiskPrediction.py
# Match RapidMiner Decision Tree EXACTLY by using RapidMiner's exported tree text as the model.
# Computes Q1–Q10 using real logic + real math from the tree's leaf distributions.

from __future__ import annotations

from pathlib import Path
import re
import pandas as pd


# =========================
# REPO-RELATIVE FILE SETTINGS (portable)
# =========================
# Assumes repo structure:
#   repo/
#     data/InsuranceData.xlsx
#     src/InsuranceRiskPrediction.py
#     outputs/
BASE_DIR = Path(__file__).resolve().parents[1]  # repo root (because this script lives in /src)
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILE = DATA_DIR / "InsuranceData.xlsx"
TRAIN_SHEET = "Issued Policies"
SCORE_SHEET = "New Applicants"

LABEL_COL_IN_EXCEL = "Insurance Category"  # NOTE: your Excel uses this exact column name


# =========================
# PASTE YOUR RAPIDMINER TREE TEXT HERE (as-is)
# (Optional improvement: store in rapidminer/rapidminer_tree_export.txt and read it instead)
# =========================
RAPIDMINER_TREE_TEXT = r"""
Tree
At Fault Accidents > 0.500
|   Age > 65.500: Moderate Risk {High Risk - Do Not Insure=0, Moderate Risk=3, Potentially High Risk=0, Low Risk=1}
|   Age ≤ 65.500
|   |   Age > 64.500
|   |   |   Number of Claims > 2: High Risk - Do Not Insure {High Risk - Do Not Insure=3, Moderate Risk=0, Potentially High Risk=0, Low Risk=0}
|   |   |   Number of Claims ≤ 2: Moderate Risk {High Risk - Do Not Insure=0, Moderate Risk=2, Potentially High Risk=0, Low Risk=1}
|   |   Age ≤ 64.500
|   |   |   At Fault Accidents > 1.500
|   |   |   |   Age > 59.500
|   |   |   |   |   Moving Violations = No
|   |   |   |   |   |   Number of Claims > 3: High Risk - Do Not Insure {High Risk - Do Not Insure=2, Moderate Risk=0, Potentially High Risk=0, Low Risk=0}
|   |   |   |   |   |   Number of Claims ≤ 3: Moderate Risk {High Risk - Do Not Insure=0, Moderate Risk=2, Potentially High Risk=0, Low Risk=1}
|   |   |   |   |   Moving Violations = Yes
|   |   |   |   |   |   Late Payments = No
|   |   |   |   |   |   |   Number of Claims > 2.500: Potentially High Risk {High Risk - Do Not Insure=0, Moderate Risk=0, Potentially High Risk=2, Low Risk=0}
|   |   |   |   |   |   |   Number of Claims ≤ 2.500: Moderate Risk {High Risk - Do Not Insure=0, Moderate Risk=5, Potentially High Risk=1, Low Risk=0}
|   |   |   |   |   |   Late Payments = Yes: Potentially High Risk {High Risk - Do Not Insure=1, Moderate Risk=0, Potentially High Risk=9, Low Risk=0}
|   |   |   |   Age ≤ 59.500
|   |   |   |   |   Moving Violations = No: Potentially High Risk {High Risk - Do Not Insure=1, Moderate Risk=0, Potentially High Risk=11, Low Risk=0}
|   |   |   |   |   Moving Violations = Yes
|   |   |   |   |   |   At Fault Accidents > 2.500: Potentially High Risk {High Risk - Do Not Insure=21, Moderate Risk=0, Potentially High Risk=48, Low Risk=0}
|   |   |   |   |   |   At Fault Accidents ≤ 2.500
|   |   |   |   |   |   |   Age > 17.500
|   |   |   |   |   |   |   |   Payment Method = Bank Transfer: Potentially High Risk {High Risk - Do Not Insure=15, Moderate Risk=0, Potentially High Risk=26, Low Risk=0}
|   |   |   |   |   |   |   |   Payment Method = Credit Card: High Risk - Do Not Insure {High Risk - Do Not Insure=9, Moderate Risk=0, Potentially High Risk=9, Low Risk=0}
|   |   |   |   |   |   |   |   Payment Method = Monthly Billing: Potentially High Risk {High Risk - Do Not Insure=7, Moderate Risk=0, Potentially High Risk=9, Low Risk=0}
|   |   |   |   |   |   |   |   Payment Method = Website Account: Potentially High Risk {High Risk - Do Not Insure=17, Moderate Risk=1, Potentially High Risk=22, Low Risk=3}
|   |   |   |   |   |   |   Age ≤ 17.500: Potentially High Risk {High Risk - Do Not Insure=0, Moderate Risk=0, Potentially High Risk=4, Low Risk=0}
|   |   |   At Fault Accidents ≤ 1.500
|   |   |   |   Age > 21.500
|   |   |   |   |   Late Payments = No
|   |   |   |   |   |   Marital Status = M
|   |   |   |   |   |   |   Comprehensive Claims = No
|   |   |   |   |   |   |   |   Age > 42: High Risk - Do Not Insure {High Risk - Do Not Insure=2, Moderate Risk=0, Potentially High Risk=1, Low Risk=2}
|   |   |   |   |   |   |   |   Age ≤ 42: Potentially High Risk {High Risk - Do Not Insure=1, Moderate Risk=4, Potentially High Risk=5, Low Risk=1}
|   |   |   |   |   |   |   Comprehensive Claims = Yes: Moderate Risk {High Risk - Do Not Insure=3, Moderate Risk=12, Potentially High Risk=2, Low Risk=2}
|   |   |   |   |   |   Marital Status = S
|   |   |   |   |   |   |   Gender = F
|   |   |   |   |   |   |   |   Number of Claims > 2.500: Moderate Risk {High Risk - Do Not Insure=0, Moderate Risk=2, Potentially High Risk=0, Low Risk=0}
|   |   |   |   |   |   |   |   Number of Claims ≤ 2.500: Potentially High Risk {High Risk - Do Not Insure=2, Moderate Risk=2, Potentially High Risk=4, Low Risk=0}
|   |   |   |   |   |   |   Gender = M: Moderate Risk {High Risk - Do Not Insure=0, Moderate Risk=10, Potentially High Risk=3, Low Risk=0}
|   |   |   |   |   Late Payments = Yes
|   |   |   |   |   |   Age > 28.500
|   |   |   |   |   |   |   Age > 34.500: Potentially High Risk {High Risk - Do Not Insure=10, Moderate Risk=4, Potentially High Risk=35, Low Risk=1}
|   |   |   |   |   |   |   Age ≤ 34.500
|   |   |   |   |   |   |   |   Gender = F: High Risk - Do Not Insure {High Risk - Do Not Insure=4, Moderate Risk=0, Potentially High Risk=0, Low Risk=0}
|   |   |   |   |   |   |   |   Gender = M: Potentially High Risk {High Risk - Do Not Insure=0, Moderate Risk=0, Potentially High Risk=2, Low Risk=1}
|   |   |   |   |   |   Age ≤ 28.500
|   |   |   |   |   |   |   Age > 24.500: Moderate Risk {High Risk - Do Not Insure=0, Moderate Risk=2, Potentially High Risk=0, Low Risk=0}
|   |   |   |   |   |   |   Age ≤ 24.500: Potentially High Risk {High Risk - Do Not Insure=0, Moderate Risk=0, Potentially High Risk=4, Low Risk=0}
|   |   |   |   Age ≤ 21.500: Potentially High Risk {High Risk - Do Not Insure=0, Moderate Risk=0, Potentially High Risk=8, Low Risk=0}
At Fault Accidents ≤ 0.500
|   Number of Claims > 3.500: Moderate Risk {High Risk - Do Not Insure=0, Moderate Risk=21, Potentially High Risk=0, Low Risk=0}
|   Number of Claims ≤ 3.500
|   |   Age > 20.500
|   |   |   Late Payments = No
|   |   |   |   Age > 25.500
|   |   |   |   |   Age > 26.500
|   |   |   |   |   |   Age > 41.500: Low Risk {High Risk - Do Not Insure=0, Moderate Risk=9, Potentially High Risk=0, Low Risk=96}
|   |   |   |   |   |   Age ≤ 41.500
|   |   |   |   |   |   |   Payment Method = Bank Transfer
|   |   |   |   |   |   |   |   Age > 38.500: Moderate Risk {High Risk - Do Not Insure=0, Moderate Risk=6, Potentially High Risk=0, Low Risk=0}
|   |   |   |   |   |   |   |   Age ≤ 38.500: Low Risk {High Risk - Do Not Insure=0, Moderate Risk=3, Potentially High Risk=0, Low Risk=9}
|   |   |   |   |   |   |   Payment Method = Credit Card
|   |   |   |   |   |   |   |   Number of Claims > 1.500: Low Risk {High Risk - Do Not Insure=0, Moderate Risk=0, Potentially High Risk=0, Low Risk=2}
|   |   |   |   |   |   |   |   Number of Claims ≤ 1.500: Moderate Risk {High Risk - Do Not Insure=0, Moderate Risk=6, Potentially High Risk=0, Low Risk=1}
|   |   |   |   |   |   |   Payment Method = Monthly Billing: Low Risk {High Risk - Do Not Insure=0, Moderate Risk=0, Potentially High Risk=0, Low Risk=11}
|   |   |   |   |   |   |   Payment Method = Website Account: Low Risk {High Risk - Do Not Insure=0, Moderate Risk=2, Potentially High Risk=0, Low Risk=21}
|   |   |   |   |   Age ≤ 26.500
|   |   |   |   |   |   Comprehensive Claims = No: Low Risk {High Risk - Do Not Insure=0, Moderate Risk=0, Potentially High Risk=0, Low Risk=2}
|   |   |   |   |   |   Comprehensive Claims = Yes: Moderate Risk {High Risk - Do Not Insure=0, Moderate Risk=2, Potentially High Risk=0, Low Risk=0}
|   |   |   |   Age ≤ 25.500
|   |   |   |   |   Comprehensive Claims = No
|   |   |   |   |   |   Number of Claims > 1.500: Low Risk {High Risk - Do Not Insure=0, Moderate Risk=0, Potentially High Risk=0, Low Risk=2}
|   |   |   |   |   |   Number of Claims ≤ 1.500: Moderate Risk {High Risk - Do Not Insure=0, Moderate Risk=4, Potentially High Risk=0, Low Risk=1}
|   |   |   |   |   Comprehensive Claims = Yes: Moderate Risk {High Risk - Do Not Insure=0, Moderate Risk=8, Potentially High Risk=0, Low Risk=0}
|   |   |   Late Payments = Yes
|   |   |   |   Age > 63.500
|   |   |   |   |   Age > 64.500: Moderate Risk {High Risk - Do Not Insure=0, Moderate Risk=2, Potentially High Risk=0, Low Risk=0}
|   |   |   |   |   Age ≤ 64.500: Low Risk {High Risk - Do Not Insure=0, Moderate Risk=0, Potentially High Risk=0, Low Risk=4}
|   |   |   |   Age ≤ 63.500
|   |   |   |   |   Age > 27.500
|   |   |   |   |   |   Age > 34.500: Moderate Risk {High Risk - Do Not Insure=0, Moderate Risk=49, Potentially High Risk=0, Low Risk=7}
|   |   |   |   |   |   Age ≤ 34.500
|   |   |   |   |   |   |   Number of Claims > 2.500: Low Risk {High Risk - Do Not Insure=0, Moderate Risk=0, Potentially High Risk=0, Low Risk=2}
|   |   |   |   |   |   |   Number of Claims ≤ 2.500: Moderate Risk {High Risk - Do Not Insure=0, Moderate Risk=3, Potentially High Risk=0, Low Risk=1}
|   |   |   |   |   Age ≤ 27.500: Moderate Risk {High Risk - Do Not Insure=0, Moderate Risk=7, Potentially High Risk=0, Low Risk=0}
|   |   Age ≤ 20.500: Moderate Risk {High Risk - Do Not Insure=0, Moderate Risk=15, Potentially High Risk=0, Low Risk=0}
""".strip()


# =========================
# PARSING + SCORING ENGINE
# =========================

CMP_RE = re.compile(r"^(?P<feat>.+?)\s*(?P<op>>|≤|=)\s*(?P<val>.+?)\s*$")
LEAF_RE = re.compile(r"^(?P<cond>.+?):\s*(?P<label>.+?)\s*\{(?P<dist>.+)\}\s*$")

CLASSES = [
    "High Risk - Do Not Insure",
    "Moderate Risk",
    "Potentially High Risk",
    "Low Risk",
]


def _depth_and_text(line: str) -> tuple[int, str]:
    # depth is the count of leading "|   " blocks
    depth = 0
    while line.startswith("|   "):
        depth += 1
        line = line[4:]
    return depth, line.strip()


def parse_tree_to_rules(tree_text: str):
    """
    Returns a list of rules:
      rule = {
        "conds": [ (feature, op, value), ... ],   # ordered path conditions
        "pred":  "Label at leaf",
        "dist":  {class_name: count, ...}         # leaf distribution counts
      }
    """
    rules = []
    stack: list[tuple[str, str, str]] = []  # conditions by depth

    for raw in tree_text.splitlines():
        raw = raw.rstrip()
        if not raw or raw.strip() == "Tree":
            continue

        depth, text = _depth_and_text(raw)

        # Leaf?
        mleaf = LEAF_RE.match(text)
        if mleaf:
            cond_text = mleaf.group("cond").strip()
            pred_label = mleaf.group("label").strip()
            dist_text = mleaf.group("dist").strip()

            # update stack to this depth
            stack = stack[:depth]
            feat, op, val = parse_condition(cond_text)
            stack.append((feat, op, val))

            dist = parse_distribution(dist_text)
            rules.append({"conds": list(stack), "pred": pred_label, "dist": dist})
            continue

        # Non-leaf condition line
        stack = stack[:depth]
        feat, op, val = parse_condition(text)
        stack.append((feat, op, val))

    return rules


def parse_condition(cond_text: str) -> tuple[str, str, str]:
    m = CMP_RE.match(cond_text)
    if not m:
        raise ValueError(f"Could not parse condition: {cond_text!r}")
    feat = m.group("feat").strip()
    op = m.group("op").strip()
    val = m.group("val").strip()
    return feat, op, val


def parse_distribution(dist_text: str) -> dict[str, int]:
    # Example: High Risk - Do Not Insure=0, Moderate Risk=3, Potentially High Risk=0, Low Risk=1
    dist = {}
    parts = [p.strip() for p in dist_text.split(",")]
    for p in parts:
        k, v = p.split("=")
        dist[k.strip()] = int(v.strip())
    # ensure all classes present
    for c in CLASSES:
        dist.setdefault(c, 0)
    return dist


def row_satisfies_condition(row: pd.Series, feat: str, op: str, val: str) -> bool:
    if feat not in row.index:
        raise KeyError(f"Feature {feat!r} not found in row columns: {list(row.index)}")

    x = row[feat]

    # handle pandas NaN
    if pd.isna(x):
        return False

    # numeric compare if op is > or ≤
    if op in (">", "≤"):
        try:
            xv = float(x)
            vv = float(val)
        except Exception:
            return False
        return (xv > vv) if op == ">" else (xv <= vv)

    # categorical equality
    if op == "=":
        # normalize strings (RapidMiner shows Yes/No, M/S, etc.)
        xs = str(x).strip()
        vs = str(val).strip()
        return xs == vs

    raise ValueError(f"Unknown operator: {op}")


def score_one(row: pd.Series, rules) -> tuple[str, dict[str, float], list[tuple[str, str, str]]]:
    """
    Returns (pred_label, confidences, matched_conditions_path)
    Confidences are computed from leaf distribution counts.
    """
    for r in rules:
        ok = True
        for feat, op, val in r["conds"]:
            if not row_satisfies_condition(row, feat, op, val):
                ok = False
                break
        if ok:
            dist = r["dist"]
            total = sum(dist.values())
            conf = {k: (dist[k] / total if total else 0.0) for k in CLASSES}
            return r["pred"], conf, r["conds"]

    raise RuntimeError("No rule matched this row. Tree may be incomplete or data contains unexpected values.")


def common_prefix(list_of_lists):
    if not list_of_lists:
        return []
    prefix = list_of_lists[0]
    for lst in list_of_lists[1:]:
        i = 0
        while i < len(prefix) and i < len(lst) and prefix[i] == lst[i]:
            i += 1
        prefix = prefix[:i]
        if not prefix:
            break
    return prefix


def feature_from_condition(cond: tuple[str, str, str]) -> str:
    return cond[0]


# =========================
# QUESTION HELPERS
# =========================

def q1_root_feature(rules) -> str:
    # root condition is the first condition in the first rule (depth 0 path)
    first_rule = rules[0]
    root = first_rule["conds"][0]
    return feature_from_condition(root)


def predict_case_from_partial(score_df: pd.DataFrame, rules, overrides: dict) -> tuple[str, dict[str, float]]:
    """
    Build a realistic case row by copying the first scoring row (so all fields exist),
    then overriding with what the question specifies.
    """
    base = score_df.iloc[0].copy()
    for k, v in overrides.items():
        base[k] = v
    pred, conf, _path = score_one(base, rules)
    return pred, conf


def q4_split_attribute_for_case(rules, partial_conds: list[tuple[str, str, str]]) -> str:
    """
    Q4 asks: which attribute is used to decide between Potentially High Risk vs High Risk - Do Not Insure
    for the described person.
    We do this by:
      - selecting all rules whose condition list begins with the partial conditions
      - keeping only leaves whose predicted label is one of those two classes
      - finding where those rule paths diverge; the next condition's feature is the split attribute.
    """
    target = {"Potentially High Risk", "High Risk - Do Not Insure"}

    matching = []
    for r in rules:
        if r["pred"] not in target:
            continue
        conds = r["conds"]
        if len(conds) < len(partial_conds):
            continue
        if conds[:len(partial_conds)] == partial_conds:
            matching.append(conds)

    if not matching:
        raise RuntimeError("No matching rules found for Q4 partial conditions. Check your partial path.")

    # find the common prefix across those matching rule paths
    cp = common_prefix(matching)

    # the next condition after the common prefix is the split that separates outcomes
    if len(cp) >= min(len(m) for m in matching):
        raise RuntimeError("Rules did not diverge after prefix; cannot identify split feature.")

    next_cond = matching[0][len(cp)]
    return feature_from_condition(next_cond)


def main():
    print("============================================================")
    print("LOADING EXCEL")
    print("============================================================")
    print("File:", FILE)

    if not FILE.exists():
        raise FileNotFoundError(
            f"Could not find Excel file at: {FILE}\n"
            f"Expected repo structure: data/InsuranceData.xlsx"
        )

    train = pd.read_excel(FILE, sheet_name=TRAIN_SHEET).dropna(how="all")
    score = pd.read_excel(FILE, sheet_name=SCORE_SHEET).dropna(how="all")

    if LABEL_COL_IN_EXCEL not in train.columns:
        raise ValueError(
            f"Label column '{LABEL_COL_IN_EXCEL}' not found. Columns: {list(train.columns)}"
        )

    print("Training shape:", train.shape)
    print("Scoring shape :", score.shape)
    print("Training columns:", list(train.columns))
    print("Scoring columns :", list(score.columns))

    print("\n============================================================")
    print("PARSING RAPIDMINER TREE")
    print("============================================================")
    rules = parse_tree_to_rules(RAPIDMINER_TREE_TEXT)
    print(f"Parsed {len(rules)} leaf rules from the RapidMiner tree.")

    print("\n============================================================")
    print("SCORING NEW APPLICANTS (exact RapidMiner tree)")
    print("============================================================")

    # Score all rows
    preds = []
    conf_rows = []

    for i in range(len(score)):
        row = score.iloc[i]
        pred, conf, _path = score_one(row, rules)
        preds.append(pred)
        conf_rows.append(conf)

    scored = score.copy()
    scored["prediction(Insurance Category)"] = preds

    # confidence columns
    for c in CLASSES:
        scored[f"confidence({c})"] = [cr[c] for cr in conf_rows]

    # Save scored file to outputs/
    out_file = OUT_DIR / "NewApplicants_with_predictions.csv"
    scored.to_csv(out_file, index=False)
    print("Saved scored predictions to:", out_file)

    # =========================================================
    # ANSWER Q1–Q10 FROM THE SCORED RESULTS
    # =========================================================
    print("\n============================================================")
    print("ANSWERS (computed from tree + data)")
    print("============================================================")

    # Q1
    q1 = q1_root_feature(rules)
    print(f"Q1) First predictive independent variable: {q1}")

    # Q2
    q2_pred, _q2_conf = predict_case_from_partial(
        score, rules, {"At Fault Accidents": 1, "Age": 66}
    )
    print(f"Q2) AtFault>=1 and Age>65 => predicted: {q2_pred}")

    # Q3
    q3_pred, _q3_conf = predict_case_from_partial(
        score, rules,
        {
            "Age": 47,
            "Gender": "M",
            "Comprehensive Claims": "No",
            "Late Payments": "No",
            "At Fault Accidents": 0,
            "Number of Claims": 1
        }
    )
    print(f"Q3) 47M; Comp=No; Late=No; AtFault=0; Claims=1 => predicted: {q3_pred}")

    # Q4
    q4_partial = [
        ("At Fault Accidents", ">", "0.500"),
        ("Age", "≤", "65.500"),
        ("Age", "≤", "64.500"),
        ("At Fault Accidents", ">", "1.500"),
        ("Age", "≤", "59.500"),
        ("Moving Violations", "=", "Yes"),
        ("At Fault Accidents", "≤", "2.500"),
        ("Age", ">", "17.500"),
    ]
    q4_attr = q4_split_attribute_for_case(rules, q4_partial)
    print(f"Q4) Attribute deciding Potentially High Risk vs High Risk - Do Not Insure: {q4_attr}")

    # Q5
    counts = scored["prediction(Insurance Category)"].value_counts()
    q5 = counts.idxmax()
    print(f"Q5) Most frequently predicted category: {q5}")

    # Q6
    q6 = int((scored["prediction(Insurance Category)"] == "Potentially High Risk").sum())
    print(f"Q6) # predicted Potentially High Risk: {q6}")

    # Q7
    high_mask = scored["prediction(Insurance Category)"] == "High Risk - Do Not Insure"
    if high_mask.any():
        q7 = float(scored.loc[high_mask, "confidence(High Risk - Do Not Insure)"].max() * 100)
        print(f"Q7) Highest confidence among High Risk - Do Not Insure predictions: {q7:.1f}%")
    else:
        print("Q7) No High Risk - Do Not Insure predictions found (unexpected).")

    # Q8
    low_mask = scored["prediction(Insurance Category)"] == "Low Risk"
    if low_mask.any():
        q8 = int((scored.loc[low_mask, "confidence(Low Risk)"] == 1.0).sum())
        print(f"Q8) # Low Risk predictions with 100% confidence: {q8}")
    else:
        print("Q8) No Low Risk predictions found (unexpected).")

    # Q9
    ph_mask = scored["prediction(Insurance Category)"] == "Potentially High Risk"
    tol = 0.0005
    target_conf = 0.634
    candidates = scored.loc[
        ph_mask & (scored["confidence(Potentially High Risk)"].sub(target_conf).abs() <= tol)
    ]
    if len(candidates) == 0:
        tol2 = 0.005
        candidates = scored.loc[
            ph_mask & (scored["confidence(Potentially High Risk)"].sub(target_conf).abs() <= tol2)
        ]

    if len(candidates) == 0:
        print("Q9) No row found with confidence(Potentially High Risk) ≈ 63.4%.")
    else:
        hr_conf = float(candidates.iloc[0]["confidence(High Risk - Do Not Insure)"] * 100)
        ph_conf = float(candidates.iloc[0]["confidence(Potentially High Risk)"] * 100)
        print(
            f"Q9) When Potentially High Risk confidence is {ph_conf:.1f}%, "
            f"High Risk - Do Not Insure confidence is {hr_conf:.1f}%"
        )

    # Q10
    mod_mask = scored["prediction(Insurance Category)"] == "Moderate Risk"
    if mod_mask.any():
        q10 = int((scored.loc[mod_mask, "confidence(Moderate Risk)"] < 0.75).sum())
        print(f"Q10) # Moderate Risk predictions with confidence < 75%: {q10}")
    else:
        print("Q10) No Moderate Risk predictions found (unexpected).")

    print("\n============================================================")
    print("PREDICTION COUNTS (sanity check)")
    print("============================================================")
    print(counts)


if __name__ == "__main__":
    main()
