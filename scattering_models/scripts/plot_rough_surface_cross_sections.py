#!/usr/bin/env python3

from pathlib import Path
import h5py
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
path = root / "rough_surface_nc" / "rough_interface_with_topography.nc"
out = root / "figures" / "rough_surface_cross_section.png"

with h5py.File(path, "r") as h:
    x = h["x"][()]
    y = h["y"][()]
    depth = h["depth_below_local_surface"][()]

j = len(y) // 2
surface = 0 * x
interface = -depth[j]

fig, ax = plt.subplots(figsize=(11, 4))
ax.fill_between(x, interface, surface, color="lightgray")
ax.plot(x, surface, color="black", lw=1.6, label="surface")
ax.plot(x, interface, color="tab:blue", lw=1.6, label="undulating layer")
ax.set_xlabel("x (m)")
ax.set_ylabel("Elevation relative to surface (m)")
ax.set_title("Rough surface model: centre-line cross-section")
ax.grid(alpha=0.25)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(out, dpi=300)
