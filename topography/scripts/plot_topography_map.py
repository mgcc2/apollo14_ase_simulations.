#!/usr/bin/env python3

from pathlib import Path
import csv
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import from_bounds
from mpl_toolkits.axes_grid1 import make_axes_locatable

root = Path(__file__).resolve().parents[1]
tiff = root / "raw" / "NAC_DTM_APOLLO14.tiff"
csv_file = root / "processed" / "A14_ASE_geophones.csv"
out = root / "figures" / "topography_map_annotated.png"

with open(csv_file) as f:
    rows = list(csv.DictReader(f))

x_all = [float(r["mapped_easting_m"]) for r in rows]
y_all = [float(r["mapped_northing_m"]) for r in rows]
left, right = min(x_all) - 120, max(x_all) + 120
bottom, top = min(y_all) - 120, max(y_all) + 120

with rasterio.open(tiff) as src:
    win = from_bounds(left, bottom, right, top, src.transform)
    z = src.read(1, window=win, masked=True)
    t = src.window_transform(win)

extent = [t.c, t.c + t.a * z.shape[1], t.f + t.e * z.shape[0], t.f]

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(z, extent=extent, origin="upper", cmap="terrain")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="4%", pad=0.12)
fig.colorbar(im, cax=cax, label="DTM elevation (m)")

for row in rows:
    x = float(row["mapped_easting_m"])
    y = float(row["mapped_northing_m"])
    name = row["geophone"]
    ax.scatter(x, y, marker="v", s=90, color="black")
    ax.text(x, y + 18, name, ha="center", va="bottom", fontsize=10, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.15))

ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.set_title("Apollo 14 geophone locations on the DTM")
fig.tight_layout()
fig.savefig(out, dpi=300)
