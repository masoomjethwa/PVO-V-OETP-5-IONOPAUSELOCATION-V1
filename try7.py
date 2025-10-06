#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from mpi4py import MPI
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
from statsmodels.nonparametric.smoothers_lowess import lowess
import csv
import plotly.express as px

# ---------------- MPI Setup ---------------- #
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ---------------- Paths ---------------- #
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "DATA"
PLOT_DIR = BASE_DIR / "plots"
LOG_FILE = PLOT_DIR / "plot_log.csv"
PLOT_DIR.mkdir(exist_ok=True)

TAB_FILE = DATA_DIR / "OETP_IONOPAUSE_LOC.TAB"

# ---------------- Helpers ---------------- #
def log_plot(plot_file: str):
    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([plot_file, datetime.now().isoformat()])

def parse_datetime(row):
    yy = int(row["DATE"]) // 1000
    doy = int(row["DATE"]) % 1000
    year = 1900 + yy
    dt = datetime(year,1,1) + timedelta(days=doy-1)
    try:
        hour, minute, second = map(int, row["PERIAPSIS_TIME"].split(":"))
        dt = dt.replace(hour=hour, minute=minute, second=second)
    except:
        pass
    return dt

def load_ionopause_data(tab_file):
    df = pd.read_csv(
        tab_file,
        sep=r"\s+",
        header=None,
        names=[
            "ORBIT","DATE","PERIAPSIS_TIME","INBOUND_SECONDS","INBOUND_TIME",
            "INBOUND_LATITUDE","INBOUND_LOCAL_SOLAR_TIME","INBOUND_ALTITUDE",
            "INBOUND_SOLAR_ZENITH_ANGLE","OUTBOUND_SECONDS","OUTBOUND_TIME",
            "OUTBOUND_LATITUDE","OUTBOUND_LOCAL_SOLAR_TIME","OUTBOUND_ALTITUDE",
            "OUTBOUND_SOLAR_ZENITH_ANGLE"
        ],
        engine="python"
    )
    df["DATETIME"] = df.apply(parse_datetime, axis=1)
    df["YEAR"] = df["DATETIME"].dt.year
    return df

def save_plot_data(df_plot, plot_number, plot_name):
    csv_file = PLOT_DIR / f"plot{plot_number}_{plot_name}.csv"
    df_plot.to_csv(csv_file, index=False)
    log_plot(csv_file.name)

def generate_plot(plot_number, plot_name, plot_func):
    """Wrapper to handle plotting + saving + CSV export"""
    try:
        res = plot_func()
        # Determine if function returns a DataFrame or (plt commands, df)
        if isinstance(res, tuple):
            df_plot = res[-1]  # assume last element is DataFrame
        elif isinstance(res, pd.DataFrame):
            df_plot = res
        else:
            df_plot = None

        # Save figure if matplotlib figure exists
        fig = plt.gcf()
        if fig.get_axes():  # non-empty figure
            plot_file = PLOT_DIR / f"plot{plot_number}_{plot_name}.png"
            plt.tight_layout()
            plt.savefig(plot_file, dpi=300)
            plt.close()
            log_plot(plot_file.name)

        if df_plot is not None:
            save_plot_data(df_plot, plot_number, plot_name)
    except Exception as e:
        print(f"[Rank {rank}] Error in plot {plot_name}: {e}")

def generate_3d_plot(df, plot_number, plot_name):
    """Interactive 3D plot using Plotly"""
    fig = px.scatter_3d(
        df, x="INBOUND_LATITUDE", y="INBOUND_ALTITUDE", z="INBOUND_SOLAR_ZENITH_ANGLE",
        color="YEAR", title=f"{plot_name} (Inbound)",
        labels={"INBOUND_LATITUDE":"Latitude (deg)","INBOUND_ALTITUDE":"Altitude (km)",
                "INBOUND_SOLAR_ZENITH_ANGLE":"Solar Zenith Angle"}
    )
    plot_file = PLOT_DIR / f"plot{plot_number}_{plot_name}_3D.html"
    fig.write_html(plot_file)
    log_plot(plot_file.name)
    save_plot_data(df[["INBOUND_LATITUDE","INBOUND_ALTITUDE","INBOUND_SOLAR_ZENITH_ANGLE","YEAR"]],
                   plot_number, f"{plot_name}_3D")

# ---------------- Plot Definitions ---------------- #
def eda_plots(df):
    """Return 20 EDA plot definitions"""
    return [
        {"name":"inbound_outbound_altitude_scatter", "func": lambda: (plt.scatter(df["DATETIME"], df["INBOUND_ALTITUDE"], s=10, c="blue", label="Inbound"),
                                                                      plt.scatter(df["DATETIME"], df["OUTBOUND_ALTITUDE"], s=10, c="red", label="Outbound"),
                                                                      plt.xlabel("Datetime"), plt.ylabel("Altitude (km)"),
                                                                      plt.title("Inbound vs Outbound Altitude Over Time"),
                                                                      plt.legend(), df[["DATETIME","INBOUND_ALTITUDE","OUTBOUND_ALTITUDE"]])},
        {"name":"inbound_outbound_latitude_scatter", "func": lambda: (plt.scatter(df["DATETIME"], df["INBOUND_LATITUDE"], s=10, c="blue", label="Inbound"),
                                                                      plt.scatter(df["DATETIME"], df["OUTBOUND_LATITUDE"], s=10, c="red", label="Outbound"),
                                                                      plt.xlabel("Datetime"), plt.ylabel("Latitude (deg)"),
                                                                      plt.title("Inbound vs Outbound Latitude Over Time"),
                                                                      plt.legend(), df[["DATETIME","INBOUND_LATITUDE","OUTBOUND_LATITUDE"]])},
        {"name":"inbound_altitude_hist","func": lambda: (plt.hist(df["INBOUND_ALTITUDE"], bins=30, color="blue"), df[["INBOUND_ALTITUDE"]])},
        {"name":"outbound_altitude_hist","func": lambda: (plt.hist(df["OUTBOUND_ALTITUDE"], bins=30, color="red"), df[["OUTBOUND_ALTITUDE"]])},
        {"name":"inbound_latitude_hist","func": lambda: (plt.hist(df["INBOUND_LATITUDE"], bins=30, color="blue"), df[["INBOUND_LATITUDE"]])},
        {"name":"outbound_latitude_hist","func": lambda: (plt.hist(df["OUTBOUND_LATITUDE"], bins=30, color="red"), df[["OUTBOUND_LATITUDE"]])},
        {"name":"inbound_altitude_kde","func": lambda: (sns.kdeplot(x=df["INBOUND_ALTITUDE"], fill=True), df[["INBOUND_ALTITUDE"]])},
        {"name":"outbound_altitude_kde","func": lambda: (sns.kdeplot(x=df["OUTBOUND_ALTITUDE"], fill=True), df[["OUTBOUND_ALTITUDE"]])},
        {"name":"inbound_vs_outbound_altitude_box","func": lambda: (plt.boxplot([df["INBOUND_ALTITUDE"], df["OUTBOUND_ALTITUDE"]], tick_labels=["Inbound","Outbound"]), df[["INBOUND_ALTITUDE","OUTBOUND_ALTITUDE"]])},
        {"name":"inbound_vs_outbound_latitude_box","func": lambda: (plt.boxplot([df["INBOUND_LATITUDE"], df["OUTBOUND_LATITUDE"]], tick_labels=["Inbound","Outbound"]), df[["INBOUND_LATITUDE","OUTBOUND_LATITUDE"]])},
        {"name":"altitude_diff_scatter","func": lambda: (plt.scatter(df["DATETIME"], df["INBOUND_ALTITUDE"]-df["OUTBOUND_ALTITUDE"], c="purple"), df[["DATETIME","INBOUND_ALTITUDE","OUTBOUND_ALTITUDE"]])},
        {"name":"altitude_vs_year","func": lambda: (plt.scatter(df["YEAR"], df["INBOUND_ALTITUDE"], c="blue"), df[["YEAR","INBOUND_ALTITUDE"]])},
        {"name":"latitude_vs_year","func": lambda: (plt.scatter(df["YEAR"], df["INBOUND_LATITUDE"], c="blue"), df[["YEAR","INBOUND_LATITUDE"]])},
        {"name":"inbound_altitude_rolling","func": lambda: (plt.plot(df["DATETIME"], df["INBOUND_ALTITUDE"].rolling(10).mean(), c="blue"), df[["DATETIME","INBOUND_ALTITUDE"]])},
        {"name":"outbound_altitude_rolling","func": lambda: (plt.plot(df["DATETIME"], df["OUTBOUND_ALTITUDE"].rolling(10).mean(), c="red"), df[["DATETIME","OUTBOUND_ALTITUDE"]])},
        {"name":"inbound_vs_outbound_altitude_density","func": lambda: (sns.kdeplot(x=df["INBOUND_ALTITUDE"], y=df["OUTBOUND_ALTITUDE"], cmap="Blues", fill=True), df[["INBOUND_ALTITUDE","OUTBOUND_ALTITUDE"]])},
        {"name":"inbound_altitude_vs_sza","func": lambda: (plt.scatter(df["INBOUND_SOLAR_ZENITH_ANGLE"], df["INBOUND_ALTITUDE"], c="blue"), df[["INBOUND_SOLAR_ZENITH_ANGLE","INBOUND_ALTITUDE"]])},
        {"name":"outbound_altitude_vs_sza","func": lambda: (plt.scatter(df["OUTBOUND_SOLAR_ZENITH_ANGLE"], df["OUTBOUND_ALTITUDE"], c="red"), df[["OUTBOUND_SOLAR_ZENITH_ANGLE","OUTBOUND_ALTITUDE"]])},
        {"name":"inbound_latitude_vs_sza","func": lambda: (plt.scatter(df["INBOUND_SOLAR_ZENITH_ANGLE"], df["INBOUND_LATITUDE"], c="blue"), df[["INBOUND_SOLAR_ZENITH_ANGLE","INBOUND_LATITUDE"]])},
        {"name":"outbound_latitude_vs_sza","func": lambda: (plt.scatter(df["OUTBOUND_SOLAR_ZENITH_ANGLE"], df["OUTBOUND_LATITUDE"], c="red"), df[["OUTBOUND_SOLAR_ZENITH_ANGLE","OUTBOUND_LATITUDE"]])},
    ]

def insight_plots(df):
    return [
        {"name":"altitude_latitude_3d","func": lambda: generate_3d_plot(df, 21, "altitude_latitude_3d")},
        {"name":"inbound_altitude_lowess","func": lambda: (plt.plot(df["DATETIME"], lowess(df["INBOUND_ALTITUDE"], df["DATETIME"].astype(np.int64), frac=0.05)[:,1], c="blue"), df[["DATETIME","INBOUND_ALTITUDE"]])},
        {"name":"outbound_altitude_lowess","func": lambda: (plt.plot(df["DATETIME"], lowess(df["OUTBOUND_ALTITUDE"], df["DATETIME"].astype(np.int64), frac=0.05)[:,1], c="red"), df[["DATETIME","OUTBOUND_ALTITUDE"]])},
        {"name":"altitude_difference_over_time","func": lambda: (plt.plot(df["DATETIME"], df["INBOUND_ALTITUDE"]-df["OUTBOUND_ALTITUDE"], c="purple"), df[["DATETIME","INBOUND_ALTITUDE","OUTBOUND_ALTITUDE"]])},
        {"name":"latitudinal_variation","func": lambda: (plt.plot(df["DATETIME"], df["INBOUND_LATITUDE"], c="blue"), df[["DATETIME","INBOUND_LATITUDE"]])},
    ]

# ---------------- Main ---------------- #
def main():
    if rank == 0:
        df = load_ionopause_data(TAB_FILE)
        comm.bcast(df, root=0)
    else:
        df = comm.bcast(None, root=0)

    all_plots = eda_plots(df) + insight_plots(df)
    
    for i, plot_def in enumerate(all_plots):
        if i % size == rank:
            generate_plot(i+1, plot_def["name"], plot_def["func"])

if __name__ == "__main__":
    main()
