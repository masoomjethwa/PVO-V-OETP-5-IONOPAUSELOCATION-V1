#!/usr/bin/env python3
# @author: mp10
# @coding assistant: TGC-06Oct2025
# pip install mpi4py pandas matplotlib statsmodels

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from mpi4py import MPI
from statsmodels.nonparametric.smoothers_lowess import lowess

# -------------------------------------------------------------------------
# Global config
# -------------------------------------------------------------------------
# BASE_DIR = Path(__file__).parent
# TAB_FILE = BASE_DIR / "OETP_IONOPAUSE_LOC.TAB"
# PLOT_DIR = BASE_DIR / "plots"
# PLOT_DIR.mkdir(exist_ok=True)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "DATA"
TAB_FILE = DATA_DIR / "OETP_IONOPAUSE_LOC.TAB"
PLOT_DIR = BASE_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# Columns as defined in PDS label
COLUMN_NAMES = [
    "ORBIT",
    "DATE_YYDOY",
    "PERIAPSIS_TIME",
    "INBOUND_SECONDS",
    "INBOUND_TIME",
    "INBOUND_LATITUDE",
    "INBOUND_LOCAL_SOLAR_TIME",
    "INBOUND_ALTITUDE",
    "INBOUND_SOLAR_ZENITH_ANGLE",
    "OUTBOUND_SECONDS",
    "OUTBOUND_TIME",
    "OUTBOUND_LATITUDE",
    "OUTBOUND_LOCAL_SOLAR_TIME",
    "OUTBOUND_ALTITUDE",
    "OUTBOUND_SOLAR_ZENITH_ANGLE",
]


# -------------------------------------------------------------------------
# Data model
# -------------------------------------------------------------------------
@dataclass
class IonopauseRecord:
    orbit: int
    datetime: datetime
    inbound_altitude: float
    outbound_altitude: float
    inbound_latitude: float
    outbound_latitude: float


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------
def parse_datetime(date_yydoy: int, periapsis_time: str) -> datetime:
    """
    Convert PDS YYDOY + HH:MM:SS into full datetime (UTC).
    Example: 78339 + 15:11:12 -> 1978-12-05T15:11:12
    """
    year = 1900 + int(str(date_yydoy)[:2])
    doy = int(str(date_yydoy)[2:])
    base_date = datetime(year, 1, 1) + timedelta(days=doy - 1)

    try:
        hh, mm, ss = map(int, periapsis_time.split(":"))
        return base_date.replace(hour=hh, minute=mm, second=ss)
    except Exception:
        return base_date


def load_ionopause_data(tab_file: Path) -> pd.DataFrame:
    """
    Load OETP_IONOPAUSE_LOC.TAB into DataFrame with a proper datetime column.
    """
    warnings.filterwarnings("ignore", category=FutureWarning)

    df = pd.read_csv(
        tab_file,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
        na_values=["", "NaN"],
    )

    df["DATETIME"] = df.apply(
        lambda row: parse_datetime(row["DATE_YYDOY"], row["PERIAPSIS_TIME"]), axis=1
    )

    return df


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------
def plot_inbound_outbound_scatter(df: pd.DataFrame, output_file: Path) -> None:
    """Scatter inbound vs outbound altitude vs latitude."""
    plt.figure(figsize=(8, 6))
    plt.scatter(
        df["INBOUND_LATITUDE"], df["INBOUND_ALTITUDE"], s=12, c="blue", alpha=0.6, label="Inbound"
    )
    plt.scatter(
        df["OUTBOUND_LATITUDE"], df["OUTBOUND_ALTITUDE"], s=12, c="red", alpha=0.6, label="Outbound"
    )
    plt.xlabel("Latitude (deg)")
    plt.ylabel("Altitude (km)")
    plt.title("Ionopause Inbound vs Outbound Crossings")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_altitude_timeseries(df: pd.DataFrame, output_file: Path) -> None:
    """Time series of altitude over datetime (color-coded inbound/outbound)."""
    plt.figure(figsize=(12, 6))
    plt.plot(df["DATETIME"], df["INBOUND_ALTITUDE"], "b.", markersize=3, label="Inbound")
    plt.plot(df["DATETIME"], df["OUTBOUND_ALTITUDE"], "r.", markersize=3, label="Outbound")
    plt.xlabel("Datetime (UTC)")
    plt.ylabel("Altitude (km)")
    plt.title("Ionopause Altitude over Time")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_altitude_trend(df: pd.DataFrame, output_file: Path) -> None:
    """Smoothed trend line using LOWESS."""
    plt.figure(figsize=(12, 6))

    for col, color, label in [
        ("INBOUND_ALTITUDE", "blue", "Inbound"),
        ("OUTBOUND_ALTITUDE", "red", "Outbound"),
    ]:
        valid = df.dropna(subset=["DATETIME", col])
        if valid.empty:
            continue
        x = valid["DATETIME"].map(datetime.toordinal)
        y = valid[col]
        smoothed = lowess(y, x, frac=0.05)
        plt.plot(
            [datetime.fromordinal(int(xx)) for xx in smoothed[:, 0]],
            smoothed[:, 1],
            color=color,
            linewidth=2,
            label=f"{label} (trend)",
        )

    plt.xlabel("Datetime (UTC)")
    plt.ylabel("Altitude (km)")
    plt.title("Smoothed Ionopause Altitude Trend (LOWESS)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


# -------------------------------------------------------------------------
# Main (MPI parallelization)
# -------------------------------------------------------------------------
def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        df = load_ionopause_data(TAB_FILE)
    else:
        df = None

    # Broadcast to all ranks
    df = comm.bcast(df, root=0)

    if rank == 0:
        scatter_path = PLOT_DIR / "ionopause_scatter.png"
        plot_inbound_outbound_scatter(df, scatter_path)
        print(f"[Rank {rank}] Saved scatter plot -> {scatter_path}")

    elif rank == 1:
        timeseries_path = PLOT_DIR / "ionopause_timeseries.png"
        plot_altitude_timeseries(df, timeseries_path)
        print(f"[Rank {rank}] Saved raw time-series -> {timeseries_path}")

    elif rank == 2:
        trend_path = PLOT_DIR / "ionopause_trend.png"
        plot_altitude_trend(df, trend_path)
        print(f"[Rank {rank}] Saved smoothed trend -> {trend_path}")

    else:
        # Extra ranks do nothing
        print(f"[Rank {rank}] Idle (no task assigned)")


if __name__ == "__main__":
    main()
