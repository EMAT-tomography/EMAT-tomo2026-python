# EMAT-tomo2026-python

Python port of the matlab tomohawk/tomography2025 electron tomography toolbox

Capabilities: loading, pre-alignment, cropping,
removal of bad projections, background removal, tilt-axis / vertical-mass-fluctuation / feature-based /
self-consistency alignment, manual alignment, cupping correction, ASTRA reconstruction,
post-processing, rotational slices and exports.

Views of tilt series and volumes use [stackview](https://github.com/haesleinhuepf/stackview); the
interactive tools are ipywidgets/ipycanvas GUIs in the notebook; reconstructions use the
[ASTRA toolbox](https://astra-toolbox.com) (CUDA).

## Requirements

numpy, scipy, scikit-image, matplotlib, astra-toolbox (CUDA), stackview, ipywidgets, ipycanvas,
ipyevents (keyboard shortcuts), tifffile, mrcfile, h5py, pandas, tqdm and rosettasciio (TIA `.emi/.ser`).

## Getting started

1. Start JupyterLab (or VS Code) with an environment that contains the required libraries and open `tomography.ipynb`.
2. The first cell adds this folder to `sys.path` (the equivalent of MATLAB's `addpath`), so the
   `rgtomo` package is importable without installing it. Optionally, `pip install -e .` in this folder
   installs it into the environment instead.
3. Run the cells from top to bottom; some sections are alternatives (e.g. the
   three ways of removing bad images) and some are optional.

A minimal reconstruction can be obtained with just:

```python
import rgtomo as rgt
series = rgt.uniloader(fname, None, 'tif')
series.data = rgt.normalize_intensities(series.data)
rec = rgt.reconstruct(series, 'sirt', 50, True, 30)
rgt.orthoslice_viewer(rec)
```

## Credits and licences

- **tomography2025**: R. Girod (UAntwerp). tomohawk, wh-tomo-tools (W. Heyvaert) and
  FastAutomatedTomography (A. Kadu et al.), EMAT, UAntwerp, for conventional processing, cleaning, and general workflow.
- The **cSAXS MATLAB package** (CXS group, Paul
  Scherrer Institut) for the VMF and self-consistency alignments routines (M. Odstrčil, et al. Opt. Express 27, 36637 (2019))
- **ASTRA toolbox** for the reconstructions (W. van Aarle et al., Opt. Express 24, 25129 (2016)).
