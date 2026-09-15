#!/usr/bin/env python3

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
import rasterio

root = Path(__file__).resolve().parents[1]
tiff = root / "raw" / "NAC_DTM_APOLLO14.tiff"
csv_file = root / "processed" / "A14_ASE_geophones.csv"
out = root / "figures" / "topography_original_vs_10cm_rms.png"

with open(csv_file) as f:
    g = {row["geophone"]: row for row in csv.DictReader(f)}

d = np.linspace(0, 91.44, 47)
f = d / 91.44
e = float(g["G3"]["corrected_easting_m"]) + f * (float(g["G1"]["corrected_easting_m"]) - float(g["G3"]["corrected_easting_m"]))
n = float(g["G3"]["corrected_northing_m"]) + f * (float(g["G1"]["corrected_northing_m"]) - float(g["G3"]["corrected_northing_m"]))

with rasterio.open(tiff) as src:
    z = np.array([v[0] for v in src.sample(zip(e, n))])

orig = z - z[-1]
k = np.ones(9) / 9
trend = np.convolve(np.pad(orig, 4, mode="edge"), k, mode="valid")
rough = orig - trend
used = trend + rough * (0.10 / np.sqrt(np.mean(rough ** 2)))
used -= used[-1]

fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True, sharey=True)
for a, y, title, note in [
    (ax[0], orig, "Original profile from raw DTM", "Directly sampled from NAC_DTM_APOLLO14.tiff"),
    (ax[1], used, "Profile used in final runs: 10 cm RMS roughness", "Same profile after applying the 10 cm RMS roughness rule"),
]:
    a.plot(d, y, color="black", lw=1.8)
    for name in ["G3", "G2", "G1"]:
        x = float(g[name]["distance_from_G3_m"])
        yy = np.interp(x, d, y)
        a.scatter(x, yy, marker="v", s=80)
        a.text(x, yy + 0.06, name, ha="center", va="bottom", fontweight="bold")
    a.set_xlim(-2, 94)
    a.set_ylabel("Elevation relative to G1 (m)")
    a.set_title(title)
    a.text(0.01, 0.06, note, transform=a.transAxes, fontsize=9)
    a.grid(alpha=0.2)

ax[1].set_xlabel("Distance from G3 (m)")
fig.tight_layout()
fig.savefig(out, dpi=300)
