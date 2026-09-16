from math import sqrt
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR / "placement_predict_50k Dataset (3)(in).csv"
PROCESSED_DATA_PATH = BASE_DIR / "placement_preprocessed.csv"

FEATURE_COLUMNS = [
    "CGPA",
    "AptitudeTestScore",
    "CodingTestScore",
    "MockInterviewScore",
]


def load_salary_dataset(data_path: str | Path | None = None) -> pd.DataFrame:
    if data_path is None:
        data_path = PROCESSED_DATA_PATH if PROCESSED_DATA_PATH.exists() else RAW_DATA_PATH

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if "Salary Package" not in df.columns:
        raise ValueError("The dataset must contain the 'Salary Package' column for linear regression.")

    if all(col in df.columns for col in FEATURE_COLUMNS):
        df = df[FEATURE_COLUMNS + ["Salary Package"]].dropna().copy()
    else:
        numeric_cols = [
            col for col in df.select_dtypes(include=["number"]).columns
            if col != "Salary Package"
        ]
        if len(numeric_cols) < 2:
            raise ValueError("Not enough numeric feature columns available for training.")
        df = df[numeric_cols + ["Salary Package"]].dropna().copy()

    return df


def train_linear_regression_model(data_path: str | Path | None = None) -> dict:
    df = load_salary_dataset(data_path)

    X = df[FEATURE_COLUMNS]
    y = df["Salary Package"]

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    root_mse = sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    sample_values = x_test.iloc[0].to_dict()
    sample_prediction = float(model.predict([[
        sample_values["CGPA"],
        sample_values["AptitudeTestScore"],
        sample_values["CodingTestScore"],
        sample_values["MockInterviewScore"],
    ]])[0])

    return {
        "model_name": "Linear Regression",
        "feature_columns": FEATURE_COLUMNS,
        "training_rows": len(df),
        "mse": float(mse),
        "rmse": float(root_mse),
        "r2_score": float(r2),
        "sample_prediction": sample_prediction,
        "sample_input": sample_values,
    }


def run_salary_prediction_demo(data_path: str | Path | None = None) -> dict:
    model_summary = train_linear_regression_model(data_path)
    return model_summary


if __name__ == "__main__":
    result = train_linear_regression_model()
    print("Model:", result["model_name"])
    print("Features:", result["feature_columns"])
    print("Training rows:", result["training_rows"])
    print("MSE:", result["mse"])
    print("RMSE:", result["rmse"])
    print("R2 Score:", result["r2_score"])
    print("Sample prediction:", result["sample_prediction"])
    print("Sample input:", result["sample_input"])

    # Plot actual vs predicted values for a quick local check.
    df = load_salary_dataset()
    X = df[FEATURE_COLUMNS]
    y = df["Salary Package"]
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Salary Package")
    plt.ylabel("Predicted Salary Package")
    plt.title("Actual vs Predicted Salary Package")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle="--")
    plt.show()