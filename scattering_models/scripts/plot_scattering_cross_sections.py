#!/usr/bin/env python3

from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
out = root / "figures" / "scattering_model_examples.png"


def load_section(path):
    with h5py.File(path, "r") as h:
        x = h["x"][()]
        y = h["y"][()]
        z = h["z"][()]
        vp = h["vp"][()]
    iy = len(y) // 2
    section = 100 * vp[:, iy, :].T
    return x, z, section


def load_stitched(path1, path2):
    x1, z1, s1 = load_section(path1)
    x2, z2, s2 = load_section(path2)
    if np.isclose(z2[0], z1[-1]):
        z2 = z2[1:]
        s2 = s2[1:, :]
    return x1, np.concatenate([z1, z2]), np.vstack([s1, s2])

items = [
    ("Global linear 25%",) + load_section(root / "scattering_nc" / "linear_25pct.nc"),
    ("Global non-linear 25%",) + load_section(root / "scattering_nc" / "nonlinear_25pct.nc"),
    ("Separate-layer linear 50%",) + load_stitched(root / "scattering_nc" / "separate_layer_linear_50pct_layer01.nc", root / "scattering_nc" / "separate_layer_linear_50pct_layer02.nc"),
    ("Separate-layer non-linear 50%",) + load_stitched(root / "scattering_nc" / "separate_layer_nonlinear_50pct_layer01.nc", root / "scattering_nc" / "separate_layer_nonlinear_50pct_layer02.nc"),
]

fig, ax = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True, constrained_layout=True)
for a, (title, x, z, s) in zip(ax.flat, items):
    im = a.imshow(s, extent=[x.min(), x.max(), z.max(), z.min()], cmap="RdBu_r", vmin=-50, vmax=50, aspect="auto")
    a.set_title(title)
    a.set_xlabel("x (m)")
    a.set_ylabel("Depth below surface (m)")

cbar = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.02)
cbar.set_label("Vp perturbation (%)")
fig.savefig(out, dpi=300)
