from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR / "placement_predict_50k Dataset (3)(in).csv"
PROCESSED_DATA_PATH = BASE_DIR / "placement_preprocessed.csv"


def load_training_data():
    if PROCESSED_DATA_PATH.exists():
        df = pd.read_csv(PROCESSED_DATA_PATH)
    else:
        df = pd.read_csv(RAW_DATA_PATH)
        df = df.drop(columns=["StudentID"], errors="ignore")

        target = "PlacementStatus"
        numeric_cols = [
            col for col in df.select_dtypes(include=["number"]).columns
            if col not in {target, "Salary Package", "IsAnomaly"}
        ]
        categorical_cols = [
            col for col in df.columns
            if col not in numeric_cols + [target, "Salary Package", "IsAnomaly"]
        ]

        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode().iloc[0])
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    if "PlacementStatus" not in df.columns:
        raise ValueError("PlacementStatus column is missing in the dataset.")

    X = df.drop(columns=["PlacementStatus"], errors="ignore")
    y = df["PlacementStatus"].astype(int)
    return X, y


def train_model():
    if LGBMClassifier is None:
        raise ImportError("lightgbm is not installed. Install it with: pip install lightgbm")

    X, y = load_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        objective="binary",
    )
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    return {
        "model_name": "LightGBM",
        "train_accuracy": float(accuracy_score(y_train, train_pred)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }


if __name__ == "__main__":
    result = train_model()
    print(result)
