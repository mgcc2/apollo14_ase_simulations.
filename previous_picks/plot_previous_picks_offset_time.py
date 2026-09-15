#!/usr/bin/env python3
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

directory = Path(__file__).resolve().parent

input_file = directory / "apollo14_previous_p_wave_picks_97.txt"
output_file = directory / "apollo14_previous_picks_offset_time.png"

# -------------------------------------------------------------------
# Read picks
# -------------------------------------------------------------------

df = pd.read_csv(
    input_file,
    comment="#",
    sep=r"\s+"
)

df["offset_m"] = pd.to_numeric(df["offset_m"], errors="coerce")
df["pick_s"] = pd.to_numeric(df["pick_s"], errors="coerce")

df = df.dropna(subset=["offset_m", "pick_s", "source"])


# -------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))

for source in df["source"].unique():
    picks = df[df["source"] == source]

    ax.scatter(
        picks["offset_m"],
        picks["pick_s"],
        s=40,
        label=source
    )

ax.set_xlabel("Source-receiver offset (m)")
ax.set_ylabel("P-wave arrival time (s)")
ax.set_title("Apollo 14 previously published P-wave picks")

ax.legend(title="Source")
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(output_file, dpi=300)

print(f"Loaded {len(df)} picks")
print(f"Saved: {output_file}")