#!/usr/bin/env python3
# @author: mp10
# @coding assistant: [TGC-06102025]
# pip install pandas

"""
Pioneer Venus Orbiter (PVO) Ionopause Crossing Loader

This script reads the PDS3 formatted fixed-width table file
(OETP_IONOPAUSE_LOC.TAB) using metadata from its label file (.LBL),
and loads the data into a pandas DataFrame with proper column names.

Data source:
PVO-V-OETP-5-IONOPAUSELOCATION-V1.0
Planetary Data System (PDS3 archive)
"""

from pathlib import Path
import pandas as pd


def load_ionopause_data(tab_path: Path) -> pd.DataFrame:
    """
    Load the Pioneer Venus Orbiter ionopause crossing table.

    Parameters
    ----------
    tab_path : Path
        Path to the .TAB file.

    Returns
    -------
    pd.DataFrame
        Parsed DataFrame with all 15 columns.
    """

    # Fixed-width field widths based on .LBL "BYTES"
    colspecs = [
        (1, 6),     # ORBIT
        (7, 12),    # DATE (YYDOY)
        (14, 22),   # PERIAPSIS_TIME
        (24, 29),   # INBOUND_SECONDS
        (34, 42),   # INBOUND_TIME
        (45, 50),   # INBOUND_LATITUDE
        (51, 55),   # INBOUND_LOCAL_SOLAR_TIME
        (56, 62),   # INBOUND_ALTITUDE
        (63, 68),   # INBOUND_SOLAR_ZENITH_ANGLE
        (73, 78),   # OUTBOUND_SECONDS
        (80, 88),   # OUTBOUND_TIME
        (90, 95),   # OUTBOUND_LATITUDE
        (96, 100),  # OUTBOUND_LOCAL_SOLAR_TIME
        (101, 107), # OUTBOUND_ALTITUDE
        (109, 114), # OUTBOUND_SOLAR_ZENITH_ANGLE
    ]

    column_names = [
        "orbit",
        "date_yydoy",
        "periapsis_time",
        "inbound_seconds",
        "inbound_time",
        "inbound_latitude",
        "inbound_local_solar_time",
        "inbound_altitude",
        "inbound_solar_zenith_angle",
        "outbound_seconds",
        "outbound_time",
        "outbound_latitude",
        "outbound_local_solar_time",
        "outbound_altitude",
        "outbound_solar_zenith_angle",
    ]

    df = pd.read_fwf(tab_path, colspecs=colspecs, header=None, names=column_names)

    # Clean up whitespace in string columns
    for col in ["periapsis_time", "inbound_time", "outbound_time"]:
        df[col] = df[col].astype(str).str.strip()

    return df


if __name__ == "__main__":
    # Adjust base path for your system
    base_dir = Path(r"C:\Users\VASCSC\Desktop\venus\PVO-V-OETP-5-IONOPAUSELOCATION-V1.0\DATA")
    tab_file = base_dir / "OETP_IONOPAUSE_LOC.TAB"

    df = load_ionopause_data(tab_file)

    print("Loaded Pioneer Venus Orbiter ionopause dataset:")
    print(df.head(10))  # show first 10 rows

    print(f"\nTotal records: {len(df)}")
