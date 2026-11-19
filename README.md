# Simulating the Apollo 14 Active Seismic Experiment with AxiSEM3D

This repository contains the Python scripts and Jupyter notebooks used to process the Apollo 14 Active Seismic Experiment observations, analyse the AxiSEM3D simulations, and generate figures for the manuscript *Simulating the Apollo 14 Active Seismic Experiment with AxiSEM3D*.

The Apollo 14 active seismic survey consisted of 21 explosive initiators deployed along a 91.44 m profile, with seismic recordings from three geophones. The analysis compares the recorded waveforms with three-dimensional numerical simulations to investigate the effects of near-surface velocity structure, topography, and seismic scattering on the observed waveforms.

The repository includes scripts for the observed Apollo 14 data, the velocity models, the scattering models, and comparisons between observed and synthetic waveforms. The corresponding AxiSEM3D simulation files are archived separately on Zenodo.

## Simulation files

The AxiSEM3D input files and synthetic seismograms for Runs 01–15 are available in the electronic supplement:

Clark, M. G. C., Nunn, C., & Panning, M. (2026). *Properties of the Lunar Regolith: Simulating the Apollo 14 Active Seismic Experiment with AxiSEM3D – Electronic Supplement – Simulation Files* (Version 1.0). Zenodo. https://doi.org/10.5281/zenodo.22758708

Each run is provided as a ZIP archive containing the input configuration, velocity and mesh files, model files, and synthetic station outputs. Vertical-component synthetic seismograms are also supplied in MiniSEED format. Wavefield animations for Shot 17 are provided separately.

## Authors

Michael G. C. Clark, Ceri Nunn, and Mark P. Panning.

## Citation

If using the scripts or notebooks in this repository, please cite the accompanying manuscript and the archived software release once its Zenodo DOI is available.
