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

## Getting started - on colab

A google colab [notebook](https://colab.research.google.com/drive/1tIl27NedC5-OZ-7DSNg_nO6hhytwprTZ?usp=sharing) is provided to quickly test the workflow.
1. Go to the notebook link. You can run the cells as is, but if you want to save your own changes, first do *File>Save a copy in Drive*
2. Connect to a GPU runtime.
3. That's it! You can run the cells and explore the results.

Some test data can be imported automatically from the notebook. To use your own data, either download the toolbox to run on your computer (see below) or upload the data to your google drive, then mount your drive in the colab environment:

```python
from google.colab import drive
drive.mount("/content/drive")
```

You can then retrieve your data by  point `fname` to the right folder in your drive:

```python
fname = "/content/drive/MyDrive/.../YourTomoData"
```

## Getting started - on your computer

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
