import marimo

__generated_with = "0.9.15"
app = marimo.App()


@app.cell
def __():
    # '%load_ext autoreload\n%autoreload 2' command supported automatically in marimo
    return


@app.cell
def __():
    import sys

    sys.path.append("/app")

    from pathlib import Path

    import geopandas as gpd
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from tqdm.notebook import tqdm

    from scripts.lmfit_model import division_model, sum_model
    return Path, division_model, gpd, np, pd, plt, sum_model, sys, tqdm


@app.cell
def __(pd, plt):
    def qh_plot(qh_df: pd.DataFrame):
        x = qh_df["q_cms"].to_numpy()
        y = qh_df["lvl_sm"].to_numpy()
        y_qh = qh_df["lvl_qh"].to_numpy()

        fig, ax = plt.subplots()
        ax.scatter(x, y, s=10, label="Кривая Q(h) по АИС")
        ax.scatter(x, y_qh, s=15, label="Уровни по кривой Q(h)", color="green")
        ax.set_xlabel("Расход, куб. м/с")
        ax.set_ylabel("Уровень, см")
        ax.legend()
        plt.close()
        return ax.get_figure()
    return (qh_plot,)


@app.cell
def __(gpd):
    initial_gauges = gpd.read_file("/app/data/Geometry/russia_gauges_full.gpkg")
    initial_gauges = initial_gauges.set_index("gauge_id")
    return (initial_gauges,)


@app.cell
def __(mo):
    mo.md("### Visualize Q(h) from lmfit model")
    return


@app.cell
def __(Path, division_model, np, pd, sum_model, tqdm):
    qh_lmfit_images = Path("data/images/qh_lmfit_full")
    qh_lmfit_images.mkdir(exist_ok=True, parents=True)
    _param_path = Path("/app/data/QhDEM/params/qh_full")
    result_folder = Path("/app/data/QhDEM/res/qh_full")
    result_folder.mkdir(exist_ok=True, parents=True)
    _param_gauges = list((i.stem.split("_") for i in Path(f"{_param_path}").glob("*.json")))
    _q_gauges = list((i.stem for i in Path("/app/data/QhDEM/HydroReady").glob("*.pkl")))
    _param_gauges = [i for i in _param_gauges if i[0] in _q_gauges]
    for _desc in tqdm(_param_gauges):
        _result = None
        (gauge_id, model_type) = _desc
        _hydro_file = pd.read_csv(f"/app/data/HydroFiles/{gauge_id}.csv", parse_dates=True, index_col="date")
        _hydro_file[_hydro_file <= 0] = np.nan
        _data_file = pd.read_pickle(f"/app/data/QhDEM/HydroReady/{gauge_id}.pkl")
        _data_file = _data_file.dropna()
        _y = _data_file["lvl_sm"].to_numpy()
        _x = _data_file["q_cms"].to_numpy()
        if model_type == "sum":
            try:
                _params = sum_model.make_params(power=1, c=1, slope=1, b=1)
                _result = sum_model.fit(_y, _params, x=_x)
            except ValueError:
                _params = sum_model.make_params(power=0, c=1, slope=1, b=1)
                _result = sum_model.fit(_y, _params, x=_x)
        elif model_type == "div":
            try:
                _params = division_model.make_params(power=1, c=1, slope=1, b=1)
                _result = division_model.fit(_y, _params, x=_x)
            except ValueError:
                _params = division_model.make_params(power=0, c=1, slope=1, b=1)
                _result = division_model.fit(_y, _params, x=_x)
        else:
            print(f"{gauge_id}")
            continue
        with open(f"{_param_path}/{gauge_id}_{model_type}.json", "r") as _f:
            _params = _result.params.load(_f)
        _hydro_file["lvl_qh"] = [_result.eval(x=i, params=_params) for i in _hydro_file["q_cms"].to_numpy()]
        _hydro_file.to_pickle(f"{result_folder}/{gauge_id}.pkl")

    return gauge_id, model_type, qh_lmfit_images, result_folder


@app.cell
def __(
    Path,
    file,
    hydro_file,
    pd,
    qh_lmfit_images,
    qh_plot,
    result_folder,
    tqdm,
):
    _pkl_files = list(Path(f"{result_folder}").glob("*.pkl"))
    for _file in tqdm(_pkl_files):
        _hydro_file = pd.read_pickle(file)
        _img = qh_plot(hydro_file)
        _img.savefig(f"{qh_lmfit_images}/{_file.stem}.png")
    return


@app.cell
def __(
    Path,
    data_file,
    desc,
    division_model,
    f,
    file,
    hydro_file,
    np,
    param_gauges,
    param_path,
    params,
    pd,
    q_gauges,
    qh_plot,
    result,
    sum_model,
    tqdm,
    x,
    y,
):
    qh_lmfit_images_1 = Path("data/images/qh_lmfit_open")
    qh_lmfit_images_1.mkdir(exist_ok=True, parents=True)
    _param_path = Path("data/params/qh_open")
    result_folder_1 = Path("data/res/qh_open/")
    result_folder_1.mkdir(exist_ok=True, parents=True)
    _param_gauges = list((i.stem.split("_") for i in Path(f"{param_path}").glob("*.json")))
    _q_gauges = list((i.stem for i in Path("data/HydroReady/").glob("*.pkl")))
    _param_gauges = [i for i in param_gauges if i[0] in q_gauges]
    for _desc in tqdm(_param_gauges):
        _result = None
        (gauge_id_1, model_type_1) = desc
        if Path(f"{result_folder_1}/{_gauge_id}.pkl").is_file():
            continue
        _hydro_file = pd.read_csv(f"data/HydroFiles/{gauge_id_1}.csv", parse_dates=True, index_col="date")
        hydro_file[hydro_file <= 0] = np.nan
        _data_file = pd.read_pickle(f"data/HydroReady/{gauge_id_1}.pkl")
        _data_file = data_file.dropna()
        _y = data_file["lvl_sm"].to_numpy()
        _x = data_file["q_cms"].to_numpy()
        if _model_type == "sum":
            try:
                _params = sum_model.make_params(power=1, c=1, slope=1, b=1)
                _result = sum_model.fit(y, params, x=x)
            except ValueError:
                _params = sum_model.make_params(power=0, c=1, slope=1, b=1)
                _result = sum_model.fit(y, params, x=x)
        elif _model_type == "div":
            try:
                _params = division_model.make_params(power=1, c=1, slope=1, b=1)
                _result = division_model.fit(y, params, x=x)
            except ValueError:
                _params = division_model.make_params(power=0, c=1, slope=0, b=1)
                _result = division_model.fit(y, params, x=x)
        else:
            print(f"{_gauge_id}")
            continue
        with open(f"{_param_path}/{_gauge_id}_{_model_type}.json", "r") as _f:
            _params = result.params.load(f)
        hydro_file["lvl_qh"] = [result.eval(x=i, params=params) for i in hydro_file["q_cms"].to_numpy()]
        _hydro_file.to_pickle(f"{result_folder_1}/{_gauge_id}.pkl")
    _pkl_files = list(Path(f"{result_folder_1}").glob("*.pkl"))
    for _file in tqdm(_pkl_files):
        _hydro_file = pd.read_pickle(file)
        _img = qh_plot(hydro_file)
        _img.savefig(f"{qh_lmfit_images_1}/{_file.stem}.png")
    return gauge_id_1, model_type_1, qh_lmfit_images_1, result_folder_1


@app.cell
def __(mo):
    mo.md("### Select for next calculations only gauges with good enough Q(h)")
    return


@app.cell
def __():
    from scripts.metrics_and_visualisations import rmse
    return (rmse,)


@app.cell
def __(Path, file, hydro_file, pd, qh_plot, rmse, tqdm):
    _pkl_files = list(Path("data/res/qh_open").glob("*.pkl"))
    reasonable_gauges = list()
    for _file in tqdm(_pkl_files):
        _hydro_file = pd.read_pickle(file)
        if rmse(y_true=_hydro_file["lvl_sm"], y_pred=_hydro_file["lvl_qh"]) < 25.0:
            reasonable_gauges.append(_file.stem)
            _img = qh_plot(hydro_file)
            _img.savefig(f"data/res/qh_open_images/{_file.stem}.png")
        else:
            continue
    return (reasonable_gauges,)


@app.cell
def __(initial_gauges, reasonable_gauges):
    gauges_for_work = initial_gauges.loc[reasonable_gauges, :]
    gauges_for_work.to_file("data/geometry/qh_gauges.gpkg")
    return (gauges_for_work,)


@app.cell
def __(pd, plt, test_file):
    def plot_q_h(df: pd.DataFrame):
        # Create the figure and axes objects
        fig, ax = plt.subplots()
        # Data for the plot
        x = test_file.index
        y1 = test_file["lvl_sm"]
        y2 = test_file["lvl_qh"]

        # Create the plot
        ax.plot(x, y1, label="Уровень, см", linestyle="-", linewidth=0.9)  # Full line (default)
        ax.plot(x, y2, label="Уровень Q(h), см", linestyle="--", alpha=0.8)  # Dashed line

        # Adding labels and title
        ax.set_xlabel("Дата, день")
        ax.set_ylabel("Уровень, см")

        # Show the legend
        ax.legend()

        return ax.get_figure()


    # Display the plot
    plt.show()
    return (plot_q_h,)


@app.cell
def __(ax):
    ax.get_figure()
    return


@app.cell
def __(
    Path,
    data_file,
    desc,
    division_model,
    f,
    hydro_file,
    np,
    param_gauges,
    param_path,
    params,
    pd,
    q_gauges,
    result,
    sum_model,
    tqdm,
    x,
    y,
):
    qh_lmfit_images_2 = Path("data/images/qh_lmfit")
    qh_lmfit_images_2.mkdir(exist_ok=True, parents=True)
    _param_path = Path("data/params/qh_open")
    result_folder_2 = Path("data/res/qh_open/")
    result_folder_2.mkdir(exist_ok=True, parents=True)
    _param_gauges = list((i.stem.split("_") for i in Path(f"{param_path}").glob("*.json")))
    _q_gauges = list((i.stem for i in Path("data/HydroReady/").glob("*.pkl")))
    _param_gauges = [i for i in param_gauges if i[0] in q_gauges]
    for _desc in tqdm(_param_gauges):
        _result = None
        (gauge_id_2, model_type_2) = desc
        _hydro_file = pd.read_csv(f"data/HydroFiles/{gauge_id_2}.csv", parse_dates=True, index_col="date")
        hydro_file[hydro_file <= 0] = np.nan
        _data_file = pd.read_pickle(f"data/HydroReady/{gauge_id_2}.pkl")
        _data_file = data_file.dropna()
        _y = data_file["lvl_sm"].to_numpy()
        _x = data_file["q_cms"].to_numpy()
        if _model_type == "sum":
            _params = sum_model.make_params(power=1, c=1, slope=1, b=1)
            _result = sum_model.fit(y, params, x=x)
        elif _model_type == "div":
            try:
                _params = division_model.make_params(power=1, c=1, slope=1, b=1)
                _result = division_model.fit(y, params, x=x)
            except ValueError:
                _params = division_model.make_params(power=0, c=1, slope=1, b=1)
                _result = division_model.fit(y, params, x=x)
        else:
            print(f"{_gauge_id}")
            continue
        with open(f"{_param_path}/{_gauge_id}_{_model_type}.json", "r") as _f:
            _params = result.params.load(f)
        hydro_file["lvl_qh"] = [result.eval(x=i, params=params) for i in hydro_file["q_cms"].to_numpy()]
        _hydro_file.to_pickle(f"{result_folder_2}/{_gauge_id}.pkl")
    return gauge_id_2, model_type_2, qh_lmfit_images_2, result_folder_2


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
