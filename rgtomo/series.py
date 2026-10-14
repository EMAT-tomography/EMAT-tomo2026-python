"""Tilt-series container.

Python port of ``TiltSeries.m`` (tomohawk).

Array convention used throughout ``rgtomo``
--------------------------------------------
``TiltSeries.data`` is a 3D array of shape ``(n_angles, n_rows, n_cols)``: one
projection image per entry along the first axis. Compared to the MATLAB toolbox,
where ``data`` is ``(n_rows, n_cols, n_angles)``, MATLAB's ``data(:, :, k)`` is
``data[k - 1]`` here. Rows run along the tilt axis (vertical), columns are
perpendicular to it.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TiltSeries:
    """Holds a tilt series with optional metadata.

    Attributes
    ----------
    data : ndarray, shape (n_angles, n_rows, n_cols)
        Projection images.
    angles : ndarray, shape (n_angles,)
        Specimen holder tilt angles in degrees.
    pixel_size : float
        Pixel size in nanometers.
    times : ndarray or None
        Acquisition times (seconds), when available.
    meta : dict
        Free-form provenance information (e.g. shifts estimated by an
        alignment step).
    """

    data: np.ndarray | None = None
    angles: np.ndarray | None = None
    pixel_size: float = 1.0
    times: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.angles is not None:
            self.angles = np.asarray(self.angles, dtype=float).ravel()
        if self.times is not None:
            self.times = np.asarray(self.times, dtype=float).ravel()

    def __setattr__(self, name, value):
        if name in ("angles", "times") and value is not None:
            value = np.asarray(value, dtype=float).ravel()
        super().__setattr__(name, value)

    @property
    def n_angles(self) -> int:
        return 0 if self.data is None else self.data.shape[0]

    @property
    def shape(self) -> tuple:
        return () if self.data is None else self.data.shape

    def copy(self) -> "TiltSeries":
        """Deep copy (data, angles and metadata are copied)."""
        return copy.deepcopy(self)

    def select(self, indices) -> "TiltSeries":
        """Return a new series with only the projections at ``indices``
        (integer indices or boolean mask along the projection axis)."""
        idx = np.asarray(indices)
        out = TiltSeries(
            data=np.ascontiguousarray(self.data[idx]),
            angles=None if self.angles is None else self.angles[idx],
            pixel_size=self.pixel_size,
            times=None if self.times is None else self.times[idx],
            meta=copy.deepcopy(self.meta),
        )
        return out

    def with_data(self, data) -> "TiltSeries":
        """Return a copy of the series with ``data`` replaced."""
        out = TiltSeries(
            data=data,
            angles=None if self.angles is None else self.angles.copy(),
            pixel_size=self.pixel_size,
            times=None if self.times is None else self.times.copy(),
            meta=copy.deepcopy(self.meta),
        )
        return out

    def __repr__(self) -> str:
        if self.data is None:
            return "TiltSeries(empty)"
        n, ny, nx = self.data.shape if self.data.ndim == 3 else (1,) + self.data.shape
        ang = ""
        if self.angles is not None and self.angles.size:
            ang = f", angles {self.angles.min():.1f}..{self.angles.max():.1f} deg"
        return (f"TiltSeries({n} projections of {ny} x {nx} px, "
                f"pixel size {self.pixel_size:.4g} nm{ang}, dtype {self.data.dtype})")


def as_series(obj, angles=None) -> TiltSeries:
    """Accept a TiltSeries or a bare 3D array and return a TiltSeries."""
    if isinstance(obj, TiltSeries):
        return obj
    return TiltSeries(data=np.asarray(obj), angles=angles)
