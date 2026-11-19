# Properties of the Lunar Regolith: Simulating the Apollo 14 Active Seismic Experiment with AxiSEM3D

This repository contains the Apollo 14 Active Seismic Experiment (A14 ASE) observations, published P-wave arrival-time picks, velocity and scattering models, topography files, and Python scripts used for the analysis and figures presented in the manuscript *Properties of the Lunar Regolith: Simulating the Apollo 14 Active Seismic Experiment with AxiSEM3D*.

The A14 ASE recorded 13 successful thumper shots at three geophones (G1, G2 and G3) along a 91.44 m profile. We use the observations and 97 previously published P-wave arrival-time picks to examine the near-surface velocity structure, and compare the recorded waveforms with full-wavefield three-dimensional (3D) simulations performed using AxiSEM3D. The simulation input files and synthetic seismograms are archived separately on Zenodo.

## Repository contents

### `real_data/`

- `apollo_14/` contains the 13 available common-shot records in SEG-Y (`.segy`) format: Shots 1, 2, 3, 4, 7, 11, 12, 13, 17, 18, 19, 20 and 21. Each record contains data from G1, G2 and G3. Shots 12 and 19 have incomplete records, as described in the manuscript.
- `apollo_16/` contains 19 additional Apollo 16 shot files, stored with `.segy.txt` filenames. These are not used in the A14 ASE simulations presented in the manuscript.

### `previous_picks/`

- `apollo14_previous_p_wave_picks_97.txt` contains 97 published P-wave arrival-time picks compiled from Kovach et al. (1971; 33 picks), Martella et al. (2022; 14 picks), Villanova et al. (2025; 39 picks) and Wang et al. (2025; 11 picks). The file includes details of the original sources and pick provenance.
- `plot_previous_picks_offset_time.py` plots the picks against source-receiver offset, grouped by study.
- `apollo14_previous_picks_offset_time.png` is the corresponding figure.

### `autocorrelation_analysis/`

- `plot_apollo14_autocorrelation_3panel.py` calculates the waveform autocorrelation for each available Apollo 14 shot at G1, G2 and G3, and identifies the first negative trough within a specified lag-time interval. The five G2 recordings made before the geophone was replanted are excluded.
- `apollo14_autocorrelation_G1_G2_G3.png` is the corresponding figure.

### `velocity_models/`

The three AxiSEM3D background velocity models are provided as `.bm` files:

- `HVM.bm` – homogeneous velocity model (HVM), with a constant P-wave velocity of 104 m s⁻¹.
- `2LVM.bm` – two-layer velocity model (2LVM), with P-wave velocities of 100 and 260 m s⁻¹ separated by an interface at 7.5 m depth.
- `LGVM.bm` – linear-gradient velocity model (LGVM), with P-wave velocity increasing from 102.2 m s⁻¹ at the surface to 883.2 m s⁻¹ at 84.5 m depth. The latter value is extrapolated beyond the depths sampled by the observed arrivals.

`plot_vel_models.py` plots the P-wave velocity, S-wave velocity and density profiles. The corresponding figures (`HVM_velocity_density.png`, `2LVM_velocity_density.png` and `LGVM_velocity_density.png`) are also included.

### `topography/`

- `raw/NAC-DTM-APOLLO14.tiff` contains the Lunar Reconnaissance Orbiter Camera (LROC) Narrow Angle Camera (NAC) Digital Terrain Model (DTM) for the Apollo 14 landing site.
- `processed/A14_ASE_geophones.csv` contains the geophone coordinates and elevations.
- `processed/topography.nc` contains the processed topography model used with AxiSEM3D.
- `scripts/plot_topography_map.py` and `scripts/plot_topography_10cm_rms.py` plot the geophone locations on the DTM and compare the original topographic profile with the profile incorporating 10 cm root-mean-square (RMS) roughness.
- `figures/` contains `topography_map_annotated.png` and `topography_original_vs_10cm_rms.png`.

### `scattering_models/`

- `scattering_nc/` contains eight NetCDF (`.nc`) files for the linear scattering (LS) and power-law scattering (PLS) models. These include maximum velocity perturbations of ±25% and ±50%, together with separate-layer LS and PLS models for the 2LVM at ±50%.
- `rough_surface_nc/rough_interface_with_topography.nc` contains the rough-interface model with topography.
- `scripts/make_scattering_nc.py` generates the scattering-model files. The same fractional perturbations are applied to P-wave and S-wave velocity, while density remains unchanged.
- `scripts/plot_scattering_cross_sections.py` and `scripts/plot_rough_surface_cross_sections.py` plot cross-sections through the scattering models and rough interface.
- `figures/` contains `scattering_model_examples.png` and `rough_surface_cross_section.png`.

### `output_figures/`

This directory contains the 13 main figures and four supplementary figures from the manuscript. Each figure is supplied as a `.png` file with its corresponding Python script (`.py`) using the same filename, for example `Figure1.py` and `Figure1.png`.

The main figures are:

- **Figure 1** – A14 ASE observed waveforms at G1, G2 and G3, with the P-wave picks of Kovach et al. (1971).
- **Figure 2** – Published P-wave arrival-time picks and travel times predicted by the 2LVM.
- **Figure 3** – Published P-wave arrival-time picks and travel times predicted by the LGVM.
- **Figure 4** – P-wave velocity, S-wave velocity and density profiles for the HVM, 2LVM and LGVM.
- **Figure 5** – Examples of the LS and PLS velocity perturbations used in the simulations.
- **Figure 6** – Comparison of the observed Shot 17 waveforms with synthetic waveforms from Runs 01–15 at G1, G2 and G3.
- **Figure 7** – Comparison of the Shot 17 observations with the preferred two-layer (Run 10) and linear-gradient (Run 15) simulations.
- **Figure 8** – Wavefield-scanning azimuthal resolution requirements (Nr) for Runs 02, 03, 06 and 10.
- **Figure 9** – Observed and synthetic waveforms at G1 across the available source-receiver offsets, comparing Runs 10 and 15.
- **Figure 10** – Observed and synthetic waveforms at G2, including enlarged views of the P-wave arrivals, comparing Runs 10 and 15.
- **Figure 11** – Vertical-displacement wavefield snapshots for Runs 10 and 15 at 0.15, 0.40 and 0.75 s after Shot 17.
- **Figure 12** – Relative waveform amplitudes and envelope peak times against source-receiver distance for the observations and preferred simulations.
- **Figure 13** – Waveforms, spectrograms and amplitude envelopes for Shot 17 at G1, G2 and G3, comparing observations with Runs 10 and 15.

The supplementary figures are:

- **Supplementary Figure 1** – Observed A14 ASE waveforms and the 97 published P-wave picks, filtered at 30–60 Hz.
- **Supplementary Figure 2** – The same observations and picks, filtered at 3–100 Hz.
- **Supplementary Figure 3** – Observed waveforms aligned on the published P-wave picks, filtered at 30–60 Hz.
- **Supplementary Figure 4** – Observed waveforms aligned on the published P-wave picks, filtered at 3–100 Hz.

The plotting scripts use the observations, model files and published picks provided in this repository, together with AxiSEM3D outputs where required. Several scripts contain local path placeholders that must be changed before running. Some scripts also refer to an Apollo 14 shot catalogue (`apollo14_ase_data_catalog.txt`) that is not included in this repository. Figure 11 requires the element-output files used to generate wavefield snapshots; these large files are not included in the Zenodo simulation archive and can be regenerated by rerunning the simulations.

## AxiSEM3D simulation files

The AxiSEM3D simulation files for Runs 01–15 are available separately on Zenodo:

Clark, M. G. C., Nunn, C., & Panning, M. (2026). *Properties of the Lunar Regolith: Simulating the Apollo 14 Active Seismic Experiment with AxiSEM3D – Electronic Supplement – Simulation Files* (Version 1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.22758708

Each run is provided as a ZIP archive containing the AxiSEM3D input configuration files (`.yaml`), velocity models (`.bm`), mesh files (`.e`), NetCDF model files (`.nc`), and original synthetic station outputs. Vertical-component (U3) synthetic seismograms have also been extracted into MiniSEED (`.mseed`) format. Wavefield animations for Shot 17 are provided separately.

The raw element-output files used to generate the wavefield animations are not provided due to their substantially larger file size. These can be generated by rerunning the simulations.

## Python scripts

The repository contains Python scripts (`.py`), rather than Jupyter notebooks. Dependencies used across the scripts include NumPy, Matplotlib, Pandas, SciPy, ObsPy, netCDF4, h5py, Rasterio and PyYAML. The scripts are provided with their associated data files and figures, but are not configured as a single automated workflow. File paths may need updating for use on another computer.

## Associated manuscript

Clark, M. G. C., Nunn, C., & Panning, M. P. *Properties of the Lunar Regolith: Simulating the Apollo 14 Active Seismic Experiment with AxiSEM3D*. Manuscript in preparation.
