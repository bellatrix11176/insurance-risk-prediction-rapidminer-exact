import pandas as pd

file = r"C:\Users\gigih\OneDrive\School\Week4\InsuranceData.xlsx"

xl = pd.ExcelFile(file)
print("Sheets:", xl.sheet_names)

for sheet in xl.sheet_names:
    df = pd.read_excel(file, sheet_name=sheet)

    print("\n" + "="*70)
    print("SHEET:", sheet)
    print("Shape (rows, cols):", df.shape)
    print("Columns:", list(df.columns))
    print("\nDtypes:\n", df.dtypes)
    print("\nMissing values (top 15):\n", df.isna().sum().sort_values(ascending=False).head(15))
    print("\nPreview (first 5 rows):\n", df.head(5))
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

file = r"C:\Users\gigih\OneDrive\School\Week4\InsuranceData.xlsx"

# Load sheets
train_df = pd.read_excel(file, sheet_name="Issued Policies")
score_df = pd.read_excel(file, sheet_name="New Applicants")

# Target (label)
target = "Insurance Category"

X = train_df.drop(columns=[target])
y = train_df[target]

# Identify categorical vs numeric columns
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

# Preprocess: one-hot encode categorical, pass numeric through
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

# Default Decision Tree (like RapidMiner default-ish)
clf = DecisionTreeClassifier(random_state=0)

model = Pipeline(steps=[
    ("prep", preprocess),
    ("tree", clf)
])

model.fit(X, y)

# Get the root split feature
tree = model.named_steps["tree"]
feature_index = tree.tree_.feature[0]  # root node feature index

# Map back to the actual column/one-hot name
feature_names = model.named_steps["prep"].get_feature_names_out()

root_feature = feature_names[feature_index]
print("Root split feature (raw):", root_feature)

# Make it human-readable (strip "cat__" / "num__" prefixes)
clean = root_feature.replace("cat__", "").replace("num__", "")
# If it’s one-hot like "Late Payments_Yes", the base variable is before the underscore
base_variable = clean.split("_")[0]
print("First predictive independent variable (base):", base_variable)
