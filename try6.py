import this
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: mp10
# @coding assistant: [TGC-DD061025]
# pip install pandas numpy matplotlib seaborn statsmodels mpi4py

from mpi4py import MPI
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
from statsmodels.nonparametric.smoothers_lowess import lowess
import csv

# ---------------------- MPI Setup ---------------------- #
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ---------------------- Paths ------------------------- #
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "DATA"
PLOT_DIR = BASE_DIR / "plots"
LOG_FILE = PLOT_DIR / "plot_log.csv"
PLOT_DIR.mkdir(exist_ok=True)

TAB_FILE = DATA_DIR / "OETP_IONOPAUSE_LOC.TAB"

# ---------------------- Helpers ---------------------- #
def log_plot(plot_name: str) -> None:
    """Log plot filename and timestamp to CSV."""
    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([plot_name, datetime.now().isoformat()])

def parse_datetime(row: pd.Series) -> datetime:
    """Convert YYDOY + PERIAPSIS_TIME to full datetime."""
    yy = int(row["DATE"]) // 1000
    doy = int(row["DATE"]) % 1000
    year = 1900 + yy
    dt = datetime(year, 1, 1) + timedelta(days=doy - 1)
    hour, minute, second = map(int, row["PERIAPSIS_TIME"].split(":"))
    return dt.replace(hour=hour, minute=minute, second=second)

def load_ionopause_data(tab_file: Path) -> pd.DataFrame:
    """Load Venus ionopause TAB data."""
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

def save_plot_data(df_plot: pd.DataFrame, plot_name: str) -> None:
    """Save underlying data of a plot to CSV."""
    csv_file = PLOT_DIR / f"Plot1_{plot_name}.csv"
    df_plot.to_csv(csv_file, index=False)
    log_plot(csv_file.name)

def generate_plot(plot_name: str, plot_func) -> None:
    """Wrapper to generate a plot, save PNG, and save underlying data."""
    plt.figure(figsize=(10,6))
    df_plot = plot_func()
    plt.tight_layout()
    plot_file = PLOT_DIR / f"Plot1_{plot_name}.png"
    plt.savefig(plot_file, dpi=300)
    plt.close()
    log_plot(plot_file.name)
    if df_plot is not None:
        save_plot_data(df_plot, plot_name)

# ---------------------- Plot Definitions ---------------------- #
def eda_plots(df: pd.DataFrame) -> list:
    """Return list of 20 advanced EDA plot definitions."""
    return [
        {"name": "inbound_outbound_altitude_scatter",
         "func": lambda: (
             plt.scatter(df["DATETIME"], df["INBOUND_ALTITUDE"], s=10, c="blue", label="Inbound"),
             plt.scatter(df["DATETIME"], df["OUTBOUND_ALTITUDE"], s=10, c="red", label="Outbound"),
             plt.xlabel("Datetime"), plt.ylabel("Altitude (km)"),
             plt.title("Inbound vs Outbound Altitude Over Time"), plt.legend(),
             df[["DATETIME","INBOUND_ALTITUDE","OUTBOUND_ALTITUDE"]]
         )},
        {"name": "inbound_outbound_latitude_scatter",
         "func": lambda: (
             plt.scatter(df["DATETIME"], df["INBOUND_LATITUDE"], s=10, c="blue", label="Inbound"),
             plt.scatter(df["DATETIME"], df["OUTBOUND_LATITUDE"], s=10, c="red", label="Outbound"),
             plt.xlabel("Datetime"), plt.ylabel("Latitude (deg)"),
             plt.title("Inbound vs Outbound Latitude Over Time"), plt.legend(),
             df[["DATETIME","INBOUND_LATITUDE","OUTBOUND_LATITUDE"]]
         )},
        {"name": "inbound_altitude_histogram",
         "func": lambda: (
             plt.hist(df["INBOUND_ALTITUDE"], bins=30, color="blue", alpha=0.7),
             plt.xlabel("Inbound Altitude (km)"), plt.ylabel("Count"),
             plt.title("Histogram of Inbound Altitude"),
             df[["INBOUND_ALTITUDE"]]
         )},
        {"name": "outbound_altitude_histogram",
         "func": lambda: (
             plt.hist(df["OUTBOUND_ALTITUDE"], bins=30, color="red", alpha=0.7),
             plt.xlabel("Outbound Altitude (km)"), plt.ylabel("Count"),
             plt.title("Histogram of Outbound Altitude"),
             df[["OUTBOUND_ALTITUDE"]]
         )},
        {"name": "inbound_latitude_histogram",
         "func": lambda: (
             plt.hist(df["INBOUND_LATITUDE"], bins=30, color="blue", alpha=0.7),
             plt.xlabel("Inbound Latitude (deg)"), plt.ylabel("Count"),
             plt.title("Histogram of Inbound Latitude"),
             df[["INBOUND_LATITUDE"]]
         )},
        {"name": "outbound_latitude_histogram",
         "func": lambda: (
             plt.hist(df["OUTBOUND_LATITUDE"], bins=30, color="red", alpha=0.7),
             plt.xlabel("Outbound Latitude (deg)"), plt.ylabel("Count"),
             plt.title("Histogram of Outbound Latitude"),
             df[["OUTBOUND_LATITUDE"]]
         )},
        {"name": "inbound_vs_outbound_altitude_boxplot",
         "func": lambda: (
             plt.boxplot([df["INBOUND_ALTITUDE"], df["OUTBOUND_ALTITUDE"]], labels=["Inbound","Outbound"]),
             plt.ylabel("Altitude (km)"), plt.title("Boxplot of Altitudes"),
             df[["INBOUND_ALTITUDE","OUTBOUND_ALTITUDE"]]
         )},
        {"name": "inbound_vs_outbound_latitude_boxplot",
         "func": lambda: (
             plt.boxplot([df["INBOUND_LATITUDE"], df["OUTBOUND_LATITUDE"]], labels=["Inbound","Outbound"]),
             plt.ylabel("Latitude (deg)"), plt.title("Boxplot of Latitudes"),
             df[["INBOUND_LATITUDE","OUTBOUND_LATITUDE"]]
         )},
        {"name": "inbound_altitude_vs_sza",
         "func": lambda: (
             plt.scatter(df["INBOUND_SOLAR_ZENITH_ANGLE"], df["INBOUND_ALTITUDE"], c="blue", s=10),
             plt.xlabel("Solar Zenith Angle (deg)"), plt.ylabel("Inbound Altitude (km)"),
             plt.title("Inbound Altitude vs Solar Zenith Angle"),
             df[["INBOUND_SOLAR_ZENITH_ANGLE","INBOUND_ALTITUDE"]]
         )},
        {"name": "outbound_altitude_vs_sza",
         "func": lambda: (
             plt.scatter(df["OUTBOUND_SOLAR_ZENITH_ANGLE"], df["OUTBOUND_ALTITUDE"], c="red", s=10),
             plt.xlabel("Solar Zenith Angle (deg)"), plt.ylabel("Outbound Altitude (km)"),
             plt.title("Outbound Altitude vs Solar Zenith Angle"),
             df[["OUTBOUND_SOLAR_ZENITH_ANGLE","OUTBOUND_ALTITUDE"]]
         )},
        # 10 more EDA plots with histograms, KDEs, time trends...
        {"name": "inbound_altitude_kde", "func": lambda: (sns.kdeplot(df["INBOUND_ALTITUDE"], fill=True), df[["INBOUND_ALTITUDE"]])},
        {"name": "outbound_altitude_kde", "func": lambda: (sns.kdeplot(df["OUTBOUND_ALTITUDE"], fill=True), df[["OUTBOUND_ALTITUDE"]])},
        {"name": "inbound_latitude_kde", "func": lambda: (sns.kdeplot(df["INBOUND_LATITUDE"], fill=True), df[["INBOUND_LATITUDE"]])},
        {"name": "outbound_latitude_kde", "func": lambda: (sns.kdeplot(df["OUTBOUND_LATITUDE"], fill=True), df[["OUTBOUND_LATITUDE"]])},
        {"name": "inbound_vs_outbound_altitude_density", "func": lambda: (sns.kdeplot(df["INBOUND_ALTITUDE"], df["OUTBOUND_ALTITUDE"], cmap="Blues"), df[["INBOUND_ALTITUDE","OUTBOUND_ALTITUDE"]])},
        {"name": "altitude_vs_year", "func": lambda: (plt.scatter(df["YEAR"], df["INBOUND_ALTITUDE"], c="blue"), df[["YEAR","INBOUND_ALTITUDE"]])},
        {"name": "latitude_vs_year", "func": lambda: (plt.scatter(df["YEAR"], df["INBOUND_LATITUDE"], c="blue"), df[["YEAR","INBOUND_LATITUDE"]])},
        {"name": "inbound_altitude_rolling", "func": lambda: (plt.plot(df["DATETIME"], df["INBOUND_ALTITUDE"].rolling(10).mean(), c="blue"), df[["DATETIME","INBOUND_ALTITUDE"]])},
        {"name": "outbound_altitude_rolling", "func": lambda: (plt.plot(df["DATETIME"], df["OUTBOUND_ALTITUDE"].rolling(10).mean(), c="red"), df[["DATETIME","OUTBOUND_ALTITUDE"]])},
        {"name": "altitude_diff_scatter", "func": lambda: (plt.scatter(df["DATETIME"], df["INBOUND_ALTITUDE"] - df["OUTBOUND_ALTITUDE"], c="purple"), df[["DATETIME","INBOUND_ALTITUDE","OUTBOUND_ALTITUDE"]])},
    ]

def insight_plots(df: pd.DataFrame) -> list:
    """Return 5 scientifically meaningful insight plots."""
    return [
        {"name": "altitude_time_lowess",
         "func": lambda: (
             plt.scatter(df["DATETIME"], df["INBOUND_ALTITUDE"], s=10, c="blue", label="Inbound"),
             plt.scatter(df["DATETIME"], df["OUTBOUND_ALTITUDE"], s=10, c="red", label="Outbound"),
             # LOWESS trend
             plt.plot(df["DATETIME"], lowess(df["INBOUND_ALTITUDE"], df.index, frac=0.1)[:,1], c="cyan"),
             plt.plot(df["DATETIME"], lowess(df["OUTBOUND_ALTITUDE"], df.index, frac=0.1)[:,1], c="magenta"),
             plt.xlabel("Datetime"), plt.ylabel("Altitude (km)"),
             plt.title("Ionopause Altitude Over Time with LOWESS Trend"), plt.legend(),
             df[["DATETIME","INBOUND_ALTITUDE","OUTBOUND_ALTITUDE"]]
         )},
        {"name": "altitude_vs_sza_insight",
         "func": lambda: (
             plt.scatter(df["INBOUND_SOLAR_ZENITH_ANGLE"], df["INBOUND_ALTITUDE"], s=10, c="blue", label="Inbound"),
             plt.scatter(df["OUTBOUND_SOLAR_ZENITH_ANGLE"], df["OUTBOUND_ALTITUDE"], s=10, c="red", label="Outbound"),
             plt.xlabel("Solar Zenith Angle (deg)"), plt.ylabel("Altitude (km)"),
             plt.title("Ionopause Altitude vs Solar Zenith Angle Insight"), plt.legend(),
             df[["INBOUND_SOLAR_ZENITH_ANGLE","INBOUND_ALTITUDE","OUTBOUND_SOLAR_ZENITH_ANGLE","OUTBOUND_ALTITUDE"]]
         )},
        {"name": "altitude_diff_histogram",
         "func": lambda: (
             plt.hist(df["INBOUND_ALTITUDE"] - df["OUTBOUND_ALTITUDE"], bins=40, color="purple"),
             plt.xlabel("Inbound - Outbound Altitude (km)"), plt.ylabel("Count"),
             plt.title("Inbound vs Outbound Altitude Difference Histogram"),
             df[["INBOUND_ALTITUDE","OUTBOUND_ALTITUDE"]]
         )},
        {"name": "inbound_box_year",
         "func": lambda: (
             sns.boxplot(x="YEAR", y="INBOUND_ALTITUDE", data=df),
             plt.title("Inbound Altitude Distribution by Year"),
             plt.xlabel("Year"), plt.ylabel("Altitude (km)"),
             df[["YEAR","INBOUND_ALTITUDE"]]
         )},
        {"name": "latitude_altitude_kde_insight",
         "func": lambda: (
             sns.kdeplot(x=df["INBOUND_LATITUDE"], y=df["INBOUND_ALTITUDE"], cmap="Blues", fill=True),
             sns.kdeplot(x=df["OUTBOUND_LATITUDE"], y=df["OUTBOUND_ALTITUDE"], cmap="Reds", fill=True),
             plt.xlabel("Latitude (deg)"), plt.ylabel("Altitude (km)"),
             plt.title("Spatial KDE of Ionopause Crossings Insight"),
             df[["INBOUND_LATITUDE","INBOUND_ALTITUDE","OUTBOUND_LATITUDE","OUTBOUND_ALTITUDE"]]
         )}
    ]

# ---------------------- Main ---------------------- #
def main() -> None:
    if rank == 0:
        df = load_ionopause_data(TAB_FILE)
        comm.bcast(df, root=0)
    else:
        df = comm.bcast(None, root=0)

    all_plots = eda_plots(df) + insight_plots(df)
    
    for i, plot_def in enumerate(all_plots):
        if i % size == rank:
            try:
                generate_plot(plot_def["name"], plot_def["func"])
                print(f"[Rank {rank}] Saved plot {plot_def['name']}")
            except Exception as e:
                print(f"[Rank {rank}] Error in plot {plot_def['name']}: {e}")

if __name__ == "__main__":
    main()
