from flask import Flask, render_template

from load_data import load_data, get_data_summary
from placement_EDA import run_eda

app = Flask(__name__)


@app.route("/")
def index():
    return render_template(
        "index.html",
        active="none"
    )


@app.route("/data-loading")
def data_loading():

    error = None
    summary = None

    try:
        df = load_data()
        summary = get_data_summary(df)

    except FileNotFoundError as e:
        error = str(e)

    except Exception as e:
        error = f"Unexpected error : {e}"

    return render_template(
        "index.html",
        active="data-loading",
        summary=summary,
        error=error
    )


@app.route("/eda")
def eda_page():

    error = None
    results = None

    try:
        results = run_eda()

    except FileNotFoundError as e:
        error = str(e)

    except Exception as e:
        error = f"Unexpected error : {e}"

    return render_template(
        "eda.html",
        active="eda",
        results=results,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)