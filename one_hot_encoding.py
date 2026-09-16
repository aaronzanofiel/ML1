from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR / "placement_predict_50k Dataset (3)(in).csv"
PROCESSED_DATA_PATH = BASE_DIR / "placement_preprocessed.csv"


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove identifier column because it does not add predictive value.
    if "StudentID" in df.columns:
        df = df.drop(columns=["StudentID"])

    # Keep target column and process feature columns.
    target_column = "PlacementStatus"

    # Fill numeric missing values with median.
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_columns:
        if col == target_column:
            continue
        if df[col].isnull().any():
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)

    # Fill categorical missing values with mode.
    categorical_columns = [
        col for col in df.columns
        if col not in numeric_columns and col != target_column
    ]
    for col in categorical_columns:
        if df[col].isnull().any():
            mode_value = df[col].mode(dropna=True)
            if not mode_value.empty:
                df[col] = df[col].fillna(mode_value.iloc[0])

    # One-hot encode categorical columns.
    if categorical_columns:
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=False)

    return df


def preprocess_and_save(
    input_path: str | Path = RAW_DATA_PATH,
    output_path: str | Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Dataset not found: {input_file}")

    df = pd.read_csv(input_file)
    processed_df = preprocess_dataframe(df)
    output_file = Path(output_path)
    processed_df.to_csv(output_file, index=False)

    print(f"Original dataset shape: {df.shape}")
    print(f"Processed dataset shape: {processed_df.shape}")
    print(f"Processed file saved to: {output_file}")
    print("Columns in processed file:")
    print(processed_df.columns.tolist())
    return processed_df


if __name__ == "__main__":
    preprocess_and_save()