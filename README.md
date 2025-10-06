Here’s a **comprehensive README** draft explaining what the code does, its components, and applications:

---

# Venus Ionopause Data Analysis and 3D Visualization

## Overview

This project processes ionopause crossing data from the **Pioneer Venus Orbiter (PVO) Electron Temperature Probe (OETP)** dataset, spanning **1978–1992**, to generate **publication-quality plots and 3D visualizations**. It leverages **Python 3.10+**, **MPI for parallel computation**, and modern data-science libraries such as **pandas, matplotlib, seaborn, and plotly**.

The final output includes **25+ plots** exploring the characteristics of inbound and outbound ionopause crossings, including **altitude, latitude, local solar time, and temporal trends**, as well as **interactive 3D visualizations** of Venus-centered ionopause crossings.

---

## Data

### Sources

* **Ionopause Data**: Orbit-by-orbit times and locations of ionopause crossings (sharp gradients in electron density) stored in `.TAB` files with corresponding PDS3 `.LBL` labels.
* **Time Range**: 1978-12-05 to 1992-10-07.
* **Fields**:

  * Orbit number, date (YYDOY format), periapsis time, inbound/outbound times and seconds, altitudes, latitudes, local solar times, and solar zenith angles.

### Directory Structure

```
PVO-V-OETP-5-IONOPAUSELOCATION-V1.0/
│
├── DATA/                 # Raw ionopause data (TAB + LBL files)
├── plots/                # 2D EDA and publication-grade plots
├── plots_3d/             # 3D Venus-centered visualizations
├── try7.py               # Main MPI parallel analysis script
└── README.md             # This documentation
```

---

## Code Components

### 1. Data Loading

* Reads `.TAB` files with **pandas**, using column specifications from `.LBL` labels.
* Converts **YYDOY + periapsis time** into full UTC `datetime`.
* Computes **radius for each ionopause crossing** relative to Venus’ radius (~6051 km).

### 2. Parallel Analysis (MPI)

* **MPI ranks** (via `mpi4py`) divide plotting tasks.
* Plots are automatically named (`Plot1_{name}.png`, `Plot2_{name}.png`, ...) with corresponding CSV files (`Plot1_{name}.csv`) to ensure traceability.
* Logging of **plot filenames and timestamps** avoids Unicode errors and keeps long runs trackable.

### 3. Exploratory Data Analysis (EDA)

* **20+ automated plots** including:

  * Histograms, KDE plots, scatter plots of altitude vs latitude
  * Boxplots of inbound vs outbound altitudes/latitudes
  * Time-series of altitudes, rolling averages, LOWESS trend lines
  * Altitude differences over time, latitude vs solar zenith angle, etc.
* **Additional 5 insight plots** highlighting:

  * Temporal trends, ionopause variability, inbound/outbound comparisons
* **Looped naming system** makes adding new plots trivial.

### 4. 3D Visualization

* **Venus-centered sphere** for reference.
* **Ionopause crossings plotted in 3D spherical coordinates**:

  * X, Y, Z computed from latitude and altitude
  * Coloring by **inbound/outbound** and **local solar time**
* Outputs:

  * `venus_ionopause_3d.png`: publication-ready figure
  * `venus_ionopause_3d_xy.csv`: CSV of 3D coordinates
* **Interactive rotation and zoom** possible in Matplotlib, with optional Plotly enhancement for web-based interactivity.

---

## Code Features

* **PEP8-compliant, modular, and type-hinted**.
* Uses `pathlib` for file handling.
* **Parallelized** with `mpi4py` for faster EDA computation.
* **Automated logging**: All plot outputs tracked in CSV.
* **Publication-ready plotting**:

  * High-resolution PNGs
  * Consistent color schemes and labeling
  * 3D Venus-centered visualizations

---

## Applications

1. **Scientific Research**

   * Study ionopause dynamics around Venus.
   * Compare inbound vs outbound crossings.
   * Analyze altitude and latitude distributions.

2. **Mission Planning**

   * Useful for future Venus missions (e.g., orbital design, instrument planning).

3. **Educational Purposes**

   * Demonstrates planetary plasma interactions.
   * Can be used in classrooms for **interactive visualization of spacecraft data**.

4. **Publication Figures**

   * High-quality 2D and 3D plots for research papers.
   * Interactive 3D plots for supplementary materials.

---

## How to Run

### Requirements

```bash
pip install pandas numpy matplotlib seaborn plotly mpi4py
```

### Run MPI Script

```bash
mpiexec -n 4 python try7.py
```

* Saves **25+ plots** in `/plots` and `/plots_3d`.
* Saves **CSV logs** with plotting timestamps and data.

---

## Recommended Workflow

1. **Inspect data**: Load `.TAB` files and examine with pandas.
2. **Run MPI script**: Generate automated EDA and insight plots.
3. **Visualize 3D plots**: Use Matplotlib or optionally Plotly for interactive figures.
4. **Analyze results**: Compare inbound/outbound crossings and temporal trends.
5. **Publish or present**: Figures are ready for research papers or presentations.

---

## Notes

* The script handles **long-term datasets** efficiently via MPI.
* Color-coding by **local solar time** helps identify **day-night effects on ionopause altitude**.
* **All plots and CSVs are automatically logged**, ensuring reproducibility and traceability.
* **Further enhancements** can include interactive Plotly dashboards or web-based 3D visualizations for public outreach.

---

This README explains the **entire workflow** of your code: loading raw Venus ionopause data, parallel EDA, trend analysis, 3D visualizations, and publication-ready output.

---

# Pioneer Venus Orbiter Ionopause Analysis (V1.0)

## Overview

This project analyzes ionopause crossing data from the **Pioneer Venus Orbiter (PVO)** Electron Temperature Probe (OETP) dataset (1978-1992). It produces **25+ publication-ready plots** including 2D exploratory data analysis (EDA) and 3D Venus-centered visualizations of inbound and outbound crossings.

The workflow is parallelized with **MPI4Py**, enabling efficient computation on multi-core machines.

---

## Data

- **TAB file**: `OETP_IONOPAUSE_LOC.TAB` – Contains orbit-by-orbit ionopause crossings.
- **LBL file**: `OETP_IONOPAUSE_LOC.LBL` – Metadata describing the tabular structure.

---

## Workflow Diagram

```mermaid
flowchart TD
    A[Raw Data: PVO OETP Ionopause TAB+LBL] --> B[Data Loading & Cleaning]
    B --> C[Compute Full UTC Datetime & Derived Metrics]
    C --> D[Parallel EDA via MPI]
    D --> E[Automated 2D Plots (Histograms, KDE, Scatter, Boxplots, Trend lines)]
    D --> F[Compute Rolling Means & LOWESS Trends]
    C --> G[3D Spherical Coordinates]
    G --> H[Venus-Centered 3D Visualization]
    E --> I[Saved PNGs & CSV Logs (/plots)]
    F --> I
    H --> J[Saved 3D Plots & Data (/plots_3d)]
    I --> K[Publication-ready Figures & Analysis]
    J --> K


