#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Input files
# ---------------------------------------------------------------------

MODELS = {
    "HVM": Path(
        "/HVM.bm"
    ),

    "2LVM": Path(
        "/2LVM.bm"
    ),

    "LGVM": Path(
        "/LGVM.bm"
    ),

    "SVM": Path(
        "SVM.bm"
    ),
}


OUTDIR = Path(
    ""
)

OUTDIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Read AxiSEM3D .bm file
# ---------------------------------------------------------------------

def read_bm(path):
    units = None
    rows = []

    with open(path) as f:
        for raw in f:
            line = raw.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if parts[0].upper() == "UNITS":
                units = parts[1].lower()
                continue

            try:
                values = [float(x) for x in parts[:6]]
            except ValueError:
                continue

            if len(values) == 6:
                rows.append(values)

    if not rows:
        raise ValueError(f"No model values found in {path}")

    data = np.asarray(rows)

    depth = data[:, 0]
    vp = data[:, 1]
    vs = data[:, 2]
    rho = data[:, 3]

    if units == "km":
        depth *= 1000.0
        vp *= 1000.0
        vs *= 1000.0
        rho *= 1000.0

    elif units != "m":
        raise ValueError(f"Unrecognised UNITS '{units}' in {path}")

    return depth, vp, vs, rho


# ---------------------------------------------------------------------
# Plot each model separately
# ---------------------------------------------------------------------

for name, path in MODELS.items():

    depth, vp, vs, rho = read_bm(path)

    fig, (ax_vel, ax_rho) = plt.subplots(
        1,
        2,
        figsize=(7, 6),
        sharey=True,
        gridspec_kw={"width_ratios": [1.25, 1]},
        constrained_layout=True,
    )

    # Velocity
    ax_vel.plot(vp, depth, lw=2, label=r"$V_P$")
    ax_vel.plot(vs, depth, lw=2, label=r"$V_S$")

    ax_vel.set_xlabel("Velocity (m/s)")
    ax_vel.set_ylabel("Depth (m)")
    ax_vel.legend(frameon=False)
    ax_vel.grid(alpha=0.25)

    # Density
    ax_rho.plot(rho, depth, lw=2)

    ax_rho.set_xlabel(r"Density (kg m$^{-3}$)")
    ax_rho.grid(alpha=0.25)

    ax_vel.set_ylim(84.5, 0)

    vmax = max(vp.max(), vs.max())
    ax_vel.set_xlim(0, vmax * 1.08)

    rmin = rho.min()
    rmax = rho.max()

    if np.isclose(rmin, rmax):
        pad = 100
    else:
        pad = max(25, 0.10 * (rmax - rmin))

    ax_rho.set_xlim(rmin - pad, rmax + pad)

    fig.suptitle(name, fontsize=14)

    png = OUTDIR / f"{name}_velocity_density.png"

    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png}")