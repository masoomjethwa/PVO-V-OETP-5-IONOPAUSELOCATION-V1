#!/usr/bin/env python3
# @author: mp10
# @coding assistant: [TGC-06102025]
# pip install pandas matplotlib

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
DATA_DIR = Path(r"C:\Users\VASCSC\Desktop\venus\PVO-V-OETP-5-IONOPAUSELOCATION-V1.0\DATA")
TAB_FILE = DATA_DIR / "OETP_IONOPAUSE_LOC.TAB"

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


# ---------------------------------------------------------------------
# Helper: Convert YYDOY + HH:MM:SS → datetime
# ---------------------------------------------------------------------
def parse_yydoy_time(yydoy: int, timestr: str) -> datetime | None:
    """
    Convert YYDOY (e.g., 78339) and time string (HH:MM:SS) into datetime.
    Assumes years are 1900 + YY. Example: 78339 -> 1978, day 339.
    Returns None if input is invalid (e.g. zeros).
    """
    try:
        year = 1900 + (yydoy // 1000)
        doy = yydoy % 1000
        base_date = datetime(year, 1, 1) + timedelta(days=doy - 1)

        if timestr.strip() == "00:00:00" and yydoy == 0:
            return None

        hh, mm, ss = map(int, timestr.split(":"))
        return base_date.replace(hour=hh, minute=mm, second=ss)
    except Exception:
        return None


# ---------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------
def load_ionopause_data(tab_file: Path) -> pd.DataFrame:
    """
    Load the OETP ionopause TAB file into a DataFrame.
    """
    df = pd.read_csv(
        tab_file,
        delim_whitespace=True,
        header=None,
        names=COLUMN_NAMES,
        na_values=["", "NaN"],
    )

    # Construct full UTC datetime from DATE_YYDOY + PERIAPSIS_TIME
    df["DATETIME_UTC"] = [
        parse_yydoy_time(yydoy, tstr) for yydoy, tstr in zip(df["DATE_YYDOY"], df["PERIAPSIS_TIME"])
    ]

    return df


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def plot_inbound_outbound_scatter(df: pd.DataFrame) -> None:
    """Scatter plot of inbound vs outbound latitudes and altitudes."""
    plt.figure(figsize=(10, 6))
    plt.scatter(df["INBOUND_LATITUDE"], df["INBOUND_ALTITUDE"],
                s=20, c="blue", alpha=0.6, label="Inbound")
    plt.scatter(df["OUTBOUND_LATITUDE"], df["OUTBOUND_ALTITUDE"],
                s=20, c="red", alpha=0.6, label="Outbound")
    plt.xlabel("Latitude (deg)")
    plt.ylabel("Altitude (km)")
    plt.title("PVO Ionopause Crossings (Inbound vs Outbound)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_altitude_timeseries(df: pd.DataFrame) -> None:
    """Time series of inbound and outbound altitudes over the mission lifetime."""
    plt.figure(figsize=(12, 6))
    plt.scatter(df["DATETIME_UTC"], df["INBOUND_ALTITUDE"],
                s=15, c="blue", alpha=0.6, label="Inbound")
    plt.scatter(df["DATETIME_UTC"], df["OUTBOUND_ALTITUDE"],
                s=15, c="red", alpha=0.6, label="Outbound")
    plt.xlabel("Date (UTC)")
    plt.ylabel("Altitude (km)")
    plt.title("PVO Ionopause Altitude vs Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    df = load_ionopause_data(TAB_FILE)

    # Quick look
    print(df.head(10)[[
        "ORBIT", "DATETIME_UTC", "INBOUND_LATITUDE", "INBOUND_ALTITUDE",
        "OUTBOUND_LATITUDE", "OUTBOUND_ALTITUDE"
    ]])

    # Plots
    plot_inbound_outbound_scatter(df)
    plot_altitude_timeseries(df)
