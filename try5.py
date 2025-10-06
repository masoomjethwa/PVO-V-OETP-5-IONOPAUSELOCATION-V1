#!/usr/bin/env python3
# @author: mp10
# @coding assistant: [TGC-DD06102025]
# pip install pandas matplotlib seaborn statsmodels mpi4py

from pathlib import Path
from typing import Any
from mpi4py import MPI
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
from datetime import datetime

# -------------------------- MPI Setup -------------------------- #
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# -------------------------- Paths ----------------------------- #
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "DATA"
PLOT_DIR = BASE_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

LOG_FILE = PLOT_DIR / "plot_log.csv"

TAB_FILE = DATA_DIR / "OETP_IONOPAUSE_LOC.TAB"

# ---------------------- Logging Utility ----------------------- #
def log_plot(filename: Path) -> None:
    """Append plot filename, rank, and timestamp to CSV log."""
    log_data = pd.DataFrame([{
        "rank": rank,
        "filename": filename.name,
        "timestamp": datetime.now().isoformat()
    }])
    if LOG_FILE.exists():
        log_data.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        log_data.to_csv(LOG_FILE, mode="w", header=True, index=False)

# ---------------------- Load & preprocess --------------------- #
def load_ionopause_data(filepath: Path) -> pd.DataFrame:
    """Load fixed-width ionopause TAB file and parse datetime."""
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=[
            "ORBIT", "DATE_YYDOY", "PERIAPSIS_TIME", "INBOUND_SECONDS", "INBOUND_TIME",
            "INBOUND_LATITUDE", "INBOUND_LOCAL_SOLAR_TIME", "INBOUND_ALTITUDE",
            "INBOUND_SOLAR_ZENITH_ANGLE", "OUTBOUND_SECONDS", "OUTBOUND_TIME",
            "OUTBOUND_LATITUDE", "OUTBOUND_LOCAL_SOLAR_TIME", "OUTBOUND_ALTITUDE",
            "OUTBOUND_SOLAR_ZENITH_ANGLE"
        ],
    )

    # Convert DATE_YYDOY + PERIAPSIS_TIME to full datetime
    def yydoy_to_datetime(yydoy: int, time_str: str) -> pd.Timestamp:
        year = 1900 + int(str(yydoy)[:2])
        doy = int(str(yydoy)[2:])
        date = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=doy - 1)
        hh, mm, ss = map(int, time_str.split(":"))
        return date + pd.Timedelta(hours=hh, minutes=mm, seconds=ss)

    df["DATETIME"] = df.apply(lambda row: yydoy_to_datetime(row.DATE_YYDOY, row.PERIAPSIS_TIME), axis=1)
    df["YEAR"] = df["DATETIME"].dt.year
    return df

# ----------------------- Rank 1 Plots ------------------------- #
def plot_time_series(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    plt.scatter(df["DATETIME"], df["INBOUND_ALTITUDE"], s=10, c="blue", label="Inbound")
    plt.scatter(df["DATETIME"], df["OUTBOUND_ALTITUDE"], s=10, c="red", label="Outbound")
    # LOWESS trend
    inbound_trend = lowess(df["INBOUND_ALTITUDE"], df["DATETIME"].map(pd.Timestamp.timestamp), frac=0.02)
    outbound_trend = lowess(df["OUTBOUND_ALTITUDE"], df["DATETIME"].map(pd.Timestamp.timestamp), frac=0.02)
    plt.plot(pd.to_datetime(inbound_trend[:, 0], unit="s"), inbound_trend[:, 1], color="blue", lw=2)
    plt.plot(pd.to_datetime(outbound_trend[:, 0], unit="s"), outbound_trend[:, 1], color="red", lw=2)
    plt.xlabel("Datetime")
    plt.ylabel("Altitude (km)")
    plt.title("Venus Ionopause: Altitude vs Datetime with Trend")
    plt.legend()
    plt.tight_layout()
    out_file = PLOT_DIR / "timeseries_altitude.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    log_plot(out_file)

# ----------------------- Rank 2 Plots ------------------------- #
def plot_inbound_outbound_scatter(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(df["INBOUND_LATITUDE"], df["INBOUND_ALTITUDE"], s=10, c="blue", label="Inbound")
    plt.scatter(df["OUTBOUND_LATITUDE"], df["OUTBOUND_ALTITUDE"], s=10, c="red", label="Outbound")
    plt.xlabel("Latitude (deg)")
    plt.ylabel("Altitude (km)")
    plt.title("Inbound vs Outbound: Altitude vs Latitude")
    plt.legend()
    plt.tight_layout()
    out_file = PLOT_DIR / "scatter_latitude_altitude.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    log_plot(out_file)

# ----------------------- Rank 3 Plots ------------------------- #
def plot_advanced_eda(df: pd.DataFrame) -> None:
    sns.set(style="whitegrid")
    
    plot_tasks = [
        {"type": "hist", "columns": ["INBOUND_ALTITUDE"], "title": "Inbound Altitude Histogram", "filename": "hist_inbound_altitude.png"},
        {"type": "hist", "columns": ["OUTBOUND_ALTITUDE"], "title": "Outbound Altitude Histogram", "filename": "hist_outbound_altitude.png"},
        {"type": "hist", "columns": ["INBOUND_LATITUDE"], "title": "Inbound Latitude Histogram", "filename": "hist_inbound_latitude.png"},
        {"type": "hist", "columns": ["OUTBOUND_LATITUDE"], "title": "Outbound Latitude Histogram", "filename": "hist_outbound_latitude.png"},
        {"type": "hist", "columns": ["INBOUND_LOCAL_SOLAR_TIME"], "title": "Inbound Local Solar Time Histogram", "filename": "hist_inbound_solar_time.png"},
        {"type": "hist", "columns": ["OUTBOUND_LOCAL_SOLAR_TIME"], "title": "Outbound Local Solar Time Histogram", "filename": "hist_outbound_solar_time.png"},
        {"type": "hist", "columns": ["INBOUND_SOLAR_ZENITH_ANGLE"], "title": "Inbound Solar Zenith Angle", "filename": "hist_inbound_zenith.png"},
        {"type": "hist", "columns": ["OUTBOUND_SOLAR_ZENITH_ANGLE"], "title": "Outbound Solar Zenith Angle", "filename": "hist_outbound_zenith.png"},
        {"type": "scatter", "x": "ORBIT", "y": "INBOUND_ALTITUDE", "color": "blue", "title": "Orbit vs Inbound Altitude", "filename": "orbit_inbound_altitude.png"},
        {"type": "scatter", "x": "ORBIT", "y": "OUTBOUND_ALTITUDE", "color": "red", "title": "Orbit vs Outbound Altitude", "filename": "orbit_outbound_altitude.png"},
        {"type": "scatter", "x": "ORBIT", "y": "INBOUND_LATITUDE", "color": "blue", "title": "Orbit vs Inbound Latitude", "filename": "orbit_inbound_latitude.png"},
        {"type": "scatter", "x": "ORBIT", "y": "OUTBOUND_LATITUDE", "color": "red", "title": "Orbit vs Outbound Latitude", "filename": "orbit_outbound_latitude.png"},
        {"type": "scatter", "x": "INBOUND_ALTITUDE", "y": "OUTBOUND_ALTITUDE", "color": "purple", "title": "Inbound vs Outbound Altitude", "filename": "inbound_outbound_altitude.png"},
        {"type": "scatter", "x": "INBOUND_LATITUDE", "y": "OUTBOUND_LATITUDE", "color": "green", "title": "Inbound vs Outbound Latitude", "filename": "inbound_outbound_latitude.png"},
        {"type": "kde", "columns": ["INBOUND_ALTITUDE"], "color": "blue", "title": "Inbound Altitude KDE", "filename": "kde_inbound_altitude.png"},
        {"type": "kde", "columns": ["OUTBOUND_ALTITUDE"], "color": "red", "title": "Outbound Altitude KDE", "filename": "kde_outbound_altitude.png"},
        {"type": "kde", "columns": ["INBOUND_LATITUDE"], "color": "blue", "title": "Inbound Latitude KDE", "filename": "kde_inbound_latitude.png"},
        {"type": "kde", "columns": ["OUTBOUND_LATITUDE"], "color": "red", "title": "Outbound Latitude KDE", "filename": "kde_outbound_latitude.png"},
        {"type": "box", "x": "YEAR", "y": "INBOUND_ALTITUDE", "title": "Inbound Altitude by Year", "filename": "box_inbound_altitude.png"},
        {"type": "box", "x": "YEAR", "y": "OUTBOUND_ALTITUDE", "title": "Outbound Altitude by Year", "filename": "box_outbound_altitude.png"},
        {"type": "pair", "columns": ["INBOUND_ALTITUDE","OUTBOUND_ALTITUDE","INBOUND_LATITUDE","OUTBOUND_LATITUDE"], "filename": "pairplot.png"},
    ]

    for task in plot_tasks:
        plt.figure(figsize=(8, 5))
        try:
            if task["type"] == "hist":
                df[task["columns"]].hist(bins=30)
                plt.suptitle(task.get("title","Histogram"))
            elif task["type"] == "scatter":
                plt.scatter(df[task["x"]], df[task["y"]], s=5, c=task.get("color", "blue"))
                plt.xlabel(task["x"])
                plt.ylabel(task["y"])
                plt.title(task.get("title","Scatter Plot"))
            elif task["type"] == "kde":
                sns.kdeplot(df[task["columns"][0]].dropna(), color=task.get("color","blue"))
                plt.title(task.get("title","KDE Plot"))
            elif task["type"] == "box":
                sns.boxplot(x=task["x"], y=task["y"], data=df)
                plt.title(task.get("title","Boxplot"))
            elif task["type"] == "pair":
                sns.pairplot(df[task["columns"]].dropna())
                out_file = PLOT_DIR / task["filename"]
                plt.savefig(out_file, dpi=300)
                plt.close()
                log_plot(out_file)
                continue

            plt.tight_layout()
            out_file = PLOT_DIR / task["filename"]
            plt.savefig(out_file, dpi=300)
            plt.close()
            log_plot(out_file)
        except Exception as e:
            print(f"[Rank {rank}] Error plotting {task.get('filename')}: {e}")

# --------------------------- Main ---------------------------- #
def main() -> None:
    if rank == 0:
        df = load_ionopause_data(TAB_FILE)
        comm.bcast(df, root=0)
    else:
        df = comm.bcast(None, root=0)

    if rank == 1:
        plot_time_series(df)
        print(f"[Rank {rank}] Saved time-series altitude plot")
    elif rank == 2:
        plot_inbound_outbound_scatter(df)
        print(f"[Rank {rank}] Saved inbound vs outbound scatter plot")
    elif rank == 3:
        plot_advanced_eda(df)
        print(f"[Rank {rank}] Saved 20+ advanced EDA plots and logged filenames")

if __name__ == "__main__":
    main()
