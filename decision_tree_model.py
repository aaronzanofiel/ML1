from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from one_hot_encoding import preprocess_dataframe

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_FILE = BASE_DIR / "placement_predict_50k Dataset (3)(in).csv"
PROCESSED_DATA_FILE = BASE_DIR / "placement_preprocessed.csv"

FEATURE_COLUMNS = [
    "CGPA",
    "AptitudeTestScore",
    "CodingTestScore",
    "MockInterviewScore",
]


def load_data(filename=None):
    if filename is None:
        filename = PROCESSED_DATA_FILE if PROCESSED_DATA_FILE.exists() else RAW_DATA_FILE

    path = Path(filename)
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path)
    target = "PlacementStatus"

    if target not in data.columns:
        raise ValueError("The dataset must contain the 'PlacementStatus' target column.")

    if path.resolve() == PROCESSED_DATA_FILE.resolve():
        feature_columns = [col for col in data.columns if col != target]
    else:
        data = preprocess_dataframe(data)
        feature_columns = [col for col in data.columns if col != target]

    df = data[feature_columns + [target]].dropna().copy()
    X = df[feature_columns]
    y = df[target].astype(int)
    return X, y


def train_decision_tree_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
    )
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    return {
        "model_name": "Decision Tree",
        "feature_columns": FEATURE_COLUMNS,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_accuracy": float(accuracy_score(y_train, train_pred)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
    }


if __name__ == "__main__":
    result = train_decision_tree_model()
    print(result)
