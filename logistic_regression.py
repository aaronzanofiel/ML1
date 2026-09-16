from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from one_hot_encoding import preprocess_dataframe

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_FILE = BASE_DIR / "placement_predict_50k Dataset (3)(in).csv"
PROCESSED_DATA_FILE = BASE_DIR / "placement_preprocessed.csv"


def load_data(filename=None):
    if filename is None:
        filename = PROCESSED_DATA_FILE if PROCESSED_DATA_FILE.exists() else RAW_DATA_FILE

    path = Path(filename)
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path)

    if "PlacementStatus" not in data.columns:
        raise ValueError("The dataset must contain the 'PlacementStatus' target column.")

    if "Salary Package" in data.columns:
        data = data.drop(columns=["Salary Package"], errors="ignore")

    if PROCESSED_DATA_FILE.exists() and path.resolve() == PROCESSED_DATA_FILE.resolve():
        feature_columns = [col for col in data.columns if col != "PlacementStatus"]
    else:
        feature_columns = [
            "CGPA",
            "AptitudeTestScore",
            "CodingTestScore",
            "MockInterviewScore",
        ]
        data = preprocess_dataframe(data)
        feature_columns = [col for col in data.columns if col != "PlacementStatus"]

    X = data[feature_columns]
    y = data["PlacementStatus"]
    cleaned = pd.concat([X, y], axis=1).dropna()
    X = cleaned[feature_columns]
    y = cleaned["PlacementStatus"]

    return X, y


def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def train_and_evaluate(X_train, y_train, X_test, y_test, name):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    print(name, "Train Accuracy:", round(train_accuracy, 4))
    print(name, "Test Accuracy:", round(test_accuracy, 4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_pred))

    return model, scaler


def train_logistic_regression_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_std, y_train)

    train_pred = model.predict(X_train_std)
    test_pred = model.predict(X_test_std)

    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    cm = confusion_matrix(y_test, test_pred)

    return {
        "model_name": "Logistic Regression",
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "confusion_matrix": cm.tolist(),
    }


def main():
    summary = train_logistic_regression_model()
    print(summary)


if __name__ == "__main__":
    main()