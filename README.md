# Apollo 14 Active Seismic Experiment Simulations with AxiSEM3D

Code, data, and models associated with the study *Simulating the Apollo 14 Active Seismic Experiment with AxiSEM3D*.

## Overview

The Apollo 14 Active Seismic Experiment (ASE) was an astronaut-operated seismic survey on the Moon. This repository supports an investigation of the recorded seismic waveforms and their interpretation using three-dimensional full-wavefield simulations with [AxiSEM3D](https://github.com/AxiSEMunity/AxiSEM3D).

The study compares observed seismic records with synthetics calculated for alternative near-surface seismic velocity structures and scattering representations, with particular attention to waveform character, amplitudes, and coda development.

## Repository contents

| Directory | Description |
| --- | --- |
| `real_data/` | Material associated with the Apollo 14 ASE observations. |
| `previous_picks/` | Previously published or compiled seismic arrival-time picks. |
| `velocity_models/` | Seismic velocity-model inputs and related material. |
| `scattering_models/` | Scattering-model inputs and related material. |
| `topography/` | Topographic inputs and related material. |
| `autocorrelation_analysis/` | Autocorrelation-related analysis files. |
| `scripts/` | Analysis and processing scripts. |
| `output_figures/` | Output figures associated with the study. |

Directory descriptions identify the intended broad role of each folder; refer to the files inside each directory for specifics.

## Scientific approach

The work examines the Apollo 14 ASE records from three geophones along a 91.44 m survey profile. Travel-time data are used to assess alternative near-surface P-wave velocity models, while AxiSEM3D simulations are used to evaluate the observed waveform shapes and coda. The investigated models include layered and depth-dependent velocity structures and heterogeneous scattering models.

## Using this repository

To obtain the files, open the repository on GitHub, select **Code**, and copy the HTTPS clone URL (or choose **Download ZIP**). The repository name currently appears with a trailing period, so copying the URL directly from GitHub avoids typing an incorrect address.

The scripts and simulation inputs in this repository are research materials, not a verified one-command reproducibility workflow. The specific software versions, dependencies, and run commands required to reproduce each result should be documented alongside the corresponding scripts and simulation directories before publication.

## Associated publication

Clark, M. G. C., Nunn, C., and Panning, M. P., *Simulating the Apollo 14 Active Seismic Experiment with AxiSEM3D*. Manuscript in preparation.

Please consult the published paper, when available, for the final methodology, model specifications, and interpretation.

## Citation and archival record

A versioned release of this repository will be archived on [Zenodo](https://zenodo.org/). Once a release DOI is available, it will be added here. For analyses tied to the publication, cite the version-specific Zenodo DOI so that the exact archived materials can be retrieved.

## Contact

Michael G. C. Clark  
Department of Earth Sciences, University of Cambridge  
GitHub: [@mgcc2](https://github.com/mgcc2)

## Licensing

No licence is asserted in this README. Check the repository for an applicable `LICENSE` file and confirm reuse permissions for any third-party observational data, models, and software before distributing or reusing materials.
