#!/usr/bin/env python3

from pathlib import Path
import hashlib
import numpy as np
from netCDF4 import Dataset

root = Path(__file__).resolve().parents[1]
out = root / "scattering_nc"
out.mkdir(exist_ok=True)

SEED = 20260908
x = np.linspace(-150, 150, 234, dtype=np.float32).astype(float)
y = x.copy()


def random_signs(z):
    h = hashlib.sha256(b"scattering")
    for axis in (x, y, z):
        h.update(np.asarray(axis, dtype="<f8").tobytes())
    seed = np.random.SeedSequence([SEED, int.from_bytes(h.digest()[:4], "little")])
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(len(x), len(y), len(z)), dtype=np.int8) * 2 - 1


def write_model(name, z, amplitude, nonlinear=False):
    z = np.asarray(z, dtype=np.float32).astype(float)
    u = (z - z[0]) / (z[-1] - z[0])
    taper = 1 - (u**(1/6) if nonlinear else u)
    field = (random_signs(z) * (amplitude * taper)[None, None, :]).astype("f4")

    with Dataset(out / name, "w", format="NETCDF4") as nc:
        for axis_name, axis in {"x": x, "y": y, "z": z}.items():
            nc.createDimension(axis_name, len(axis))
            var = nc.createVariable(axis_name, "f8", (axis_name,))
            var[:] = axis
            var.units = "m"

        for var_name, data in {"vp": field, "vs": field, "rho": np.zeros_like(field)}.items():
            var = nc.createVariable(
                var_name, "f4", ("x", "y", "z"),
                zlib=True, complevel=4, fill_value=False
            )
            var[:] = data
            var.units = "1"

        nc.master_seed = SEED
        nc.random_structure = "independent bimodal sign at every x,y,z grid node"
        nc.taper = "1-u**(1/6)" if nonlinear else "1-u"
        nc.layer_top_m = float(z[0])
        nc.layer_bottom_m = float(z[-1])
        nc.amplitude_fraction = amplitude


z_global = np.linspace(0, 84.5, 51, dtype=np.float32)
z_upper = np.linspace(0, 7.5, 16, dtype=np.float32)
z_lower = np.linspace(7.5, 84.5, 155, dtype=np.float32)

write_model("linear_25pct.nc", z_global, 0.25)
write_model("linear_50pct.nc", z_global, 0.50)
write_model("nonlinear_25pct.nc", z_global, 0.25, True)
write_model("nonlinear_50pct.nc", z_global, 0.50, True)

write_model("separate_layer_linear_50pct_layer01.nc", z_upper, 0.50)
write_model("separate_layer_linear_50pct_layer02.nc", z_lower, 0.50)
write_model("separate_layer_nonlinear_50pct_layer01.nc", z_upper, 0.50, True)
write_model("separate_layer_nonlinear_50pct_layer02.nc", z_lower, 0.50, True)
