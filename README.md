# Properties of the Lunar Regolith: Simulating the Apollo 14 Active Seismic Experiment with AxiSEM3D

This repository contains the Apollo 14 Active Seismic Experiment (A14 ASE) observations, published P-wave arrival-time picks, velocity and scattering models, topography files, and Python scripts used in the analysis accompanying the manuscript *Properties of the Lunar Regolith: Simulating the Apollo 14 Active Seismic Experiment with AxiSEM3D*.

The A14 ASE recorded 13 successful thumper shots at three geophones (G1, G2 and G3) along a 91.44 m profile. We use the recorded waveforms and 97 previously published P-wave arrival-time picks to examine the near-surface velocity structure, and compare the observations with full-wavefield three-dimensional (3D) simulations performed using AxiSEM3D. The simulation files and synthetic waveforms are archived separately on Zenodo.

## Repository contents

### `real_data/`

- `apollo_14/` contains the 13 available common-shot records in SEG-Y format (`.segy`): Shots 1, 2, 3, 4, 7, 11, 12, 13, 17, 18, 19, 20 and 21. Each record contains data from G1, G2 and G3. The records for Shots 12 and 19 are incomplete, as described in the manuscript.
- `apollo_16/` contains 19 additional Apollo 16 shot files, stored with `.segy.txt` filenames. These are not used in the A14 ASE simulations described in the manuscript.

### `previous_picks/`

- `apollo14_previous_p_wave_picks_97.txt` contains 97 published P-wave arrival-time picks compiled from Kovach et al. (1971; 33 picks), Martella et al. (2022; 14 picks), Villanova et al. (2025; 39 picks) and Wang et al. (2025; 11 picks).
- `plot_previous_picks_offset_time.py` plots the picks against source-receiver offset, grouped by study.
- `apollo14_previous_picks_offset_time.png` is the resulting figure.

### `autocorrelation_analysis/`

- `plot_apollo14_autocorrelation_3panel.py` calculates the normalised waveform autocorrelation for each available Apollo 14 shot at G1, G2 and G3. It identifies the first negative trough within a specified lag-time interval and plots the results for the three geophones. The five G2 recordings made before the geophone was replanted are excluded.
- `apollo14_autocorrelation_G1_G2_G3.png` is the resulting figure.

### `velocity_models/`

The AxiSEM3D background velocity models are supplied as `.bm` files:

- `HVM.bm` – homogeneous velocity model (HVM), with a constant P-wave velocity of 104 m s⁻¹.
- `2LVM.bm` – two-layer velocity model (2LVM), with P-wave velocities of 100 and 260 m s⁻¹ separated by an interface at 7.5 m depth.
- `LGVM.bm` – linear-gradient velocity model (LGVM), with P-wave velocity increasing from 102.2 m s⁻¹ at the surface to 883.2 m s⁻¹ at 84.5 m depth. The latter value is an extrapolation of the fitted gradient beyond the depth sampled by the observed arrivals.

`plot_vel_models.py` plots the P-wave velocity, S-wave velocity and density profiles. The corresponding figures (`HVM_velocity_density.png`, `2LVM_velocity_density.png` and `LGVM_velocity_density.png`) are also included.

### `topography/`

- `raw/NAC-DTM-APOLLO14.tiff` contains the Lunar Reconnaissance Orbiter Camera (LROC) Narrow Angle Camera (NAC) Digital Terrain Model (DTM) for the Apollo 14 landing site.
- `processed/A14_ASE_geophones.csv` contains the mapped and corrected geophone coordinates and elevations.
- `processed/topography.nc` contains the processed topography model used with AxiSEM3D.
- `scripts/plot_topography_map.py` and `scripts/plot_topography_10cm_rms.py` plot the geophone locations on the DTM and compare the original profile with the profile incorporating 10 cm root-mean-square (RMS) roughness.
- `figures/` contains `topography_map_annotated.png` and `topography_original_vs_10cm_rms.png`.

### `scattering_models/`

- `scattering_nc/` contains eight NetCDF (`.nc`) scattering models. These include linear scattering (LS) and power-law scattering (PLS) with maximum velocity perturbations of ±25% and ±50%, together with separate-layer LS and PLS models for the 2LVM at ±50%.
- `rough_surface_nc/rough_interface_with_topography.nc` contains the rough-interface model with topography.
- `scripts/make_scattering_nc.py` generates the eight scattering-model files. The perturbations are applied to both P-wave and S-wave velocity, with density unchanged.
- `scripts/plot_scattering_cross_sections.py` and `scripts/plot_rough_surface_cross_sections.py` plot cross-sections through the scattering distributions and rough interface.
- `figures/` contains `scattering_model_examples.png` and `rough_surface_cross_section.png`.

### `output_figures/`

This directory is currently empty and is retained for additional output figures.

## AxiSEM3D simulation files

The complete AxiSEM3D files for Runs 01–15 are available in a separate Zenodo archive:

Clark, M. G. C., Nunn, C., & Panning, M. (2026). *Properties of the Lunar Regolith: Simulating the Apollo 14 Active Seismic Experiment with AxiSEM3D – Electronic Supplement – Simulation Files* (Version 1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.22758708

Each run is supplied as a ZIP archive containing the AxiSEM3D input configuration files (`.yaml`), velocity models (`.bm`), mesh files (`.e`), NetCDF model files (`.nc`), and original synthetic station outputs. The vertical-component (U3) synthetic seismograms are also provided in MiniSEED (`.mseed`) format. Wavefield animations for Shot 17 are provided separately. The complete simulation outputs are not duplicated in this GitHub repository.

## Python scripts

The scripts use NumPy, Matplotlib, Pandas, SciPy, ObsPy, netCDF4, h5py and Rasterio. They are provided as individual Python files (`.py`), rather than Jupyter notebooks. Some file paths in the scripts require checking or updating for a different local directory structure. The repository does not currently contain a single script that regenerates every figure in the manuscript.

## Associated manuscript

Clark, M. G. C., Nunn, C., & Panning, M. P. *Properties of the Lunar Regolith: Simulating the Apollo 14 Active Seismic Experiment with AxiSEM3D*. Manuscript in preparation.
