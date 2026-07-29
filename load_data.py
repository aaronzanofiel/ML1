
import os
from idlelib.pyshell import usage_msg

import pandas as pd

DATA_PATH = r"C:\Users\AARON\PycharmProjects\MachineLearning\placement_predict_50k Dataset (3)(in).csv"

def load_data(path : str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    return df

def get_data_summary(df: pd.DataFrame) ->  dict:
    summary = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": {col: int(df[col].isnull().sum()) for col in df.columns},
        "preview": df.head(10).to_dict( orient="records"),
    }

    return summary
if __name__ == "__main__":
    data = load_data()
    print(get_data_summary(data))