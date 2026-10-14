"""rgtomo: Python port of the rg-tomo-matlab electron tomography toolbox.

Typical use (see ``tomography.ipynb``)::

    import rgtomo as rgt
    series = rgt.uniloader("FullStack_ali.tif", ftype="tif")
    series.data = rgt.normalize_intensities(series.data)
    ...
    rec = rgt.reconstruct(series, "sirt", 50, True, 30)
    rgt.orthoslice_viewer(rec)

Conventions
-----------
* A tilt series is a :class:`TiltSeries` with ``data`` of shape
  ``(n_angles, n_rows, n_cols)`` (MATLAB: ``(n_rows, n_cols, n_angles)``),
  angles in degrees and the pixel size in nm.
* Reconstructions are ``(n_rows, n_cols, n_cols)``: slices along the tilt axis.
* Indices are 0-based, as shown by the stackview sliders.
* Interactive tools return a GUI object; get the output with ``gui.result()``.
"""

from .series import TiltSeries
from ._matlab import graythresh, rescale
from .io import (emd2stack, emi2stack, get_series_names, load_angles, read_emd,
                 read_emi, read_fasttomo, read_mat_series, read_rec, read_tif_series,
                 tomo_dir, uniloader, write_angles_csv, write_rec, write_tiff)
from .preprocessing import (apply_roi, auto_crop_rect, background_subtract,
                            crop_stack, find_edges_stack, gaussfilter_slice,
                            imresize_stack, normalize_intensities,
                            remove_bkg_median, rot90_stack)
from .reconstruction import bp, fp, has_cuda, reconstruct
from .alignment import (align_proj_com, align_stack_features, align_stack_xcorr,
                        apply_shifts_2d, apply_tilt_axis, csaxs_vmf,
                        find_shifts_2d_xcorr)
from .consistency import align_tomo_consistency
from .cleaning import (fasttomo_keepone, generate_clean_series,
                       generate_clean_tilt_series, select_images, sharpness)
from .postprocessing import (align_z_pca, cupping_correction, gaussian_filter_3d,
                             intensity_range, median_filter_3d, rotational_slices)
from .viewers import (curtain_viewer, orthoslice_viewer, side_by_side_viewer,
                      slice_viewer)
from .gui import (align_stack_manual_gui, align_tilt_axis_gui, crop_stack_gui,
                  cupping_correction_gui, select_images_gui)

__version__ = "0.1.0"
