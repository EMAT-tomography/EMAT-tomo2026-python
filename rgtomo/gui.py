"""Interactive notebook GUIs (ipywidgets + ipycanvas), replacing the MATLAB
GUIDE/uifigure tools of the toolbox:

========================  =========================================
MATLAB                    rgtomo
========================  =========================================
``crop_stack``            :func:`crop_stack_gui`
``select_images_gui_2``   :func:`select_images_gui`
``align_tilt_axis_gui``   :func:`align_tilt_axis_gui`
``align_stack_manual``    :func:`align_stack_manual_gui`
``cupping_correction_gui`` :func:`cupping_correction_gui`
========================  =========================================

Jupyter cannot block a cell while a GUI is open (MATLAB ``uiwait``), so each
function returns a GUI object that displays itself; once you are happy with
the settings, get the output in the next cell with ``gui.result()``.
``result()`` always reflects the current state of the GUI.
"""

from __future__ import annotations

import numpy as np

from .alignment import apply_tilt_axis
from .postprocessing import cupping_correction, intensity_range
from .preprocessing import apply_roi, auto_crop_rect
from .reconstruction import reconstruct
from .series import TiltSeries

__all__ = ["crop_stack_gui", "select_images_gui", "align_tilt_axis_gui",
           "align_stack_manual_gui", "cupping_correction_gui",
           "CropStackGUI", "SelectImagesGUI", "TiltAxisGUI", "ManualAlignGUI",
           "CuppingCorrectionGUI"]

# MATLAB lines() colours used by the tilt axis GUI
_BLUE, _ORANGE, _GREEN = "#0072BD", "#EDB120", "#77AC30"


# =========================================================================== helpers
def _arr(x):
    return np.asarray(x.data if isinstance(x, TiltSeries) else x)


def _to_rgb(img, vmin=None, vmax=None):
    img = np.asarray(img, dtype=np.float32)
    if vmin is None:
        vmin = float(np.nanmin(img))
    if vmax is None:
        vmax = float(np.nanmax(img))
    if vmax <= vmin:
        vmax = vmin + 1e-12
    u8 = (np.clip((img - vmin) / (vmax - vmin), 0, 1) * 255).astype(np.uint8)
    return np.repeat(u8[..., None], 3, axis=-1)


class _ImageCanvas:
    """Two-layer ipycanvas (image + overlay) with coordinates in image pixels.

    The last image and the overlay shapes are kept and redrawn whenever a
    layer reports ``client_ready``: some front ends (Google Colab's widget
    manager) lose draw commands sent before the canvas exists in the browser,
    such as an image drawn right after the canvas is created.
    """

    def __init__(self, shape, display_size=420):
        from ipycanvas import MultiCanvas
        import ipywidgets as w
        ny, nx = shape
        self.shape = (ny, nx)
        self.zoom = display_size / max(ny, nx)
        self.width = max(1, int(round(nx * self.zoom)))
        self.height = max(1, int(round(ny * self.zoom)))
        self.mc = MultiCanvas(2, width=self.width, height=self.height)
        self.mc.layout = w.Layout(width=f"{self.width}px", height=f"{self.height}px")
        self.image_layer, self.overlay = self.mc[0], self.mc[1]
        self._rgb = None
        self._shapes = []
        # MultiCanvas.on_client_ready only watches the top layer: register both
        self.image_layer.on_client_ready(self._draw_image)
        self.overlay.on_client_ready(self._draw_overlay)

    def _resample(self, img):
        ny, nx = self.shape
        if self.zoom < 1:
            from skimage.transform import resize
            return resize(img, (self.height, self.width) + img.shape[2:], order=1,
                          anti_aliasing=True, preserve_range=True).astype(img.dtype)
        r = np.clip(((np.arange(self.height) + 0.5) / self.zoom).astype(int), 0, ny - 1)
        c = np.clip(((np.arange(self.width) + 0.5) / self.zoom).astype(int), 0, nx - 1)
        return img[r][:, c]

    def set_image(self, img, vmin=None, vmax=None):
        img = np.asarray(img)
        rgb = img if (img.ndim == 3 and img.shape[-1] == 3 and img.dtype == np.uint8) else _to_rgb(img, vmin, vmax)
        self._rgb = np.ascontiguousarray(self._resample(rgb))
        self._draw_image()

    def _draw_image(self):
        if self._rgb is not None:
            self.image_layer.put_image_data(self._rgb, 0, 0)

    def clear_overlay(self):
        self._shapes = []
        self.overlay.clear()

    def line(self, x0, y0, x1, y1, color="red", width=2):
        """Line between pixel centres (x = column, y = row)."""
        z = self.zoom
        self._add_shape(("line", ((x0 + 0.5) * z, (y0 + 0.5) * z, (x1 + 0.5) * z, (y1 + 0.5) * z), color, width))

    def rect(self, x0, y0, w, h, color="yellow", width=2):
        """Rectangle covering the pixels x0..x0+w-1, y0..y0+h-1."""
        z = self.zoom
        self._add_shape(("rect", (x0 * z, y0 * z, w * z, h * z), color, width))

    def _add_shape(self, shape):
        self._shapes.append(shape)
        self._stroke(shape)

    def _stroke(self, shape):
        kind, coords, color, width = shape
        self.overlay.stroke_style = color
        self.overlay.line_width = width
        if kind == "line":
            self.overlay.stroke_line(*coords)
        else:
            self.overlay.stroke_rect(*coords)

    def _draw_overlay(self):
        from ipycanvas import hold_canvas
        with hold_canvas(self.overlay):
            self.overlay.clear()
            for shape in self._shapes:
                self._stroke(shape)

    def to_image(self, x, y):
        return x / self.zoom, y / self.zoom


class _GUIBase:
    widget = None

    def _ipython_display_(self, **kwargs):
        # display() works with ipywidgets 7 (e.g. Google Colab), which has no
        # _repr_mimebundle_ on widgets, and with ipywidgets 8
        from IPython.display import display
        display(self.widget)

    def display(self):
        from IPython.display import display
        display(self.widget)


def _help(text):
    import ipywidgets as w
    return w.HTML(f"<div style='font-size:12px;color:#666;max-width:360px'>{text}</div>")


# =========================================================================== crop
class CropStackGUI(_GUIBase):
    """Rectangular ROI on the average (or maximum) projection, see :func:`crop_stack_gui`."""

    def __init__(self, series, rect=None, display_size=480):
        import ipywidgets as w
        self.series = series
        data = _arr(series)
        self.ny, self.nx = data.shape[-2:]
        self._proj = {"Average": data.mean(axis=0), "Maximum": data.max(axis=0)}
        self.canvas = _ImageCanvas((self.ny, self.nx), display_size)
        self._auto = auto_crop_rect(data)
        x0, y0, ww, hh = rect if rect is not None else self._auto
        self.proj_toggle = w.ToggleButtons(options=["Average", "Maximum"], value="Average",
                                           description="Projection")
        self.x_slider = w.IntRangeSlider(value=[x0, x0 + ww], min=0, max=self.nx, step=1,
                                         description="X (cols)", continuous_update=True,
                                         layout=w.Layout(width="360px"))
        self.y_slider = w.IntRangeSlider(value=[y0, y0 + hh], min=0, max=self.ny, step=1,
                                         description="Y (rows)", continuous_update=True,
                                         layout=w.Layout(width="360px"))
        self.info = w.HTML()
        self.auto_btn = w.Button(description="Auto ROI", icon="magic")
        self.full_btn = w.Button(description="Full image", icon="expand")
        self.proj_toggle.observe(lambda c: self._draw_image(), names="value")
        self.x_slider.observe(lambda c: self._draw_overlay(), names="value")
        self.y_slider.observe(lambda c: self._draw_overlay(), names="value")
        self.auto_btn.on_click(lambda b: self._set_rect(self._auto))
        self.full_btn.on_click(lambda b: self._set_rect((0, 0, self.nx, self.ny)))
        self._drag = None
        self.canvas.mc.on_mouse_down(self._on_down)
        self.canvas.mc.on_mouse_move(self._on_move)
        self.canvas.mc.on_mouse_up(self._on_up)
        controls = w.VBox([self.proj_toggle, self.x_slider, self.y_slider, self.info,
                           w.HBox([self.auto_btn, self.full_btn]),
                           _help("Drag on the image to draw a new ROI or use the sliders. "
                                 "Then run the next cell: <code>series = gui.result()</code>.")])
        self.widget = w.HBox([self.canvas.mc, controls])
        self._draw_image()

    @property
    def rect(self):
        """(x0, y0, width, height) in 0-based pixels."""
        x0, x1 = self.x_slider.value
        y0, y1 = self.y_slider.value
        return (int(x0), int(y0), max(1, int(x1 - x0)), max(1, int(y1 - y0)))

    def _set_rect(self, rect):
        x0, y0, ww, hh = rect
        self.x_slider.value = [x0, x0 + ww]
        self.y_slider.value = [y0, y0 + hh]

    def _draw_image(self):
        img = self._proj[self.proj_toggle.value]
        self.canvas.set_image(img)
        self._draw_overlay()

    def _draw_overlay(self, tmp=None):
        from ipycanvas import hold_canvas
        x0, y0, ww, hh = tmp if tmp is not None else self.rect
        with hold_canvas(self.canvas.overlay):
            self.canvas.clear_overlay()
            self.canvas.rect(x0, y0, ww, hh, "yellow", 2)
        self.info.value = f"ROI: x={x0}, y={y0}, size {ww} x {hh} px"

    def _on_down(self, x, y):
        self._drag = self.canvas.to_image(x, y)

    def _drag_rect(self, x, y):
        xa, ya = self._drag
        xb, yb = self.canvas.to_image(x, y)
        x0, x1 = sorted([xa, xb])
        y0, y1 = sorted([ya, yb])
        x0, y0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
        x1, y1 = min(self.nx, int(np.ceil(x1))), min(self.ny, int(np.ceil(y1)))
        return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))

    def _on_move(self, x, y):
        if self._drag is not None:
            self._draw_overlay(self._drag_rect(x, y))

    def _on_up(self, x, y):
        if self._drag is not None:
            r = self._drag_rect(x, y)
            self._drag = None
            if r[2] > 2 and r[3] > 2:
                self._set_rect(r)
            else:
                self._draw_overlay()

    def result(self):
        """The stack (or TiltSeries) cropped to the current ROI."""
        return apply_roi(self.series, self.rect)


def crop_stack_gui(series, rect=None, display_size=480):
    """Interactive cropping (``crop_stack``): shows the average or maximum
    projection with a rectangular ROI initialised around the largest object.
    Get the cropped series with ``gui.result()``."""
    return CropStackGUI(series, rect, display_size)


# =========================================================================== select images
class SelectImagesGUI(_GUIBase):
    """Step through the projections and keep/remove each one, see :func:`select_images_gui`."""

    def __init__(self, series, display_size=384):
        import ipywidgets as w
        self.series = series
        self.data = _arr(series)
        self.angles = series.angles if isinstance(series, TiltSeries) else None
        self.n = self.data.shape[0]
        self.decisions = [None] * self.n
        self.idx = 0
        self._vmin, self._vmax = float(np.nanmin(self.data)), float(np.nanmax(self.data))
        self.canvas = _ImageCanvas(self.data.shape[-2:], display_size)
        self.title = w.HTML()
        self.keep_btn = w.Button(description="Keep", button_style="success", icon="check")
        self.remove_btn = w.Button(description="Remove", button_style="danger", icon="times")
        self.back_btn = w.Button(description="Back", icon="arrow-left")
        self.status = w.HTML()
        self.keep_btn.on_click(lambda b: self._decide(True))
        self.remove_btn.on_click(lambda b: self._decide(False))
        self.back_btn.on_click(lambda b: self._go(self.idx - 1))
        self.widget = w.VBox([self.title, self.canvas.mc,
                              w.HBox([self.remove_btn, self.keep_btn, self.back_btn]), self.status,
                              _help("Projections not reviewed are kept. Get the result with "
                                    "<code>gui.result()</code>.")])
        self._go(0)

    def _go(self, idx):
        self.idx = int(np.clip(idx, 0, self.n - 1))
        self.canvas.set_image(self.data[self.idx], self._vmin, self._vmax)
        ang = f" &mdash; angle {self.angles[self.idx]:.1f}&deg;" if self.angles is not None else ""
        state = {None: "", True: " (kept)", False: " (removed)"}[self.decisions[self.idx]]
        self.title.value = f"<b>Image {self.idx} / {self.n - 1}{ang}{state}</b>"
        kept = sum(d is True for d in self.decisions)
        removed = sum(d is False for d in self.decisions)
        self.status.value = (f"kept {kept}, removed {removed}, not reviewed {self.n - kept - removed}"
                             + (f"<br>removed: {self.excluded}" if removed else ""))

    def _decide(self, keep):
        self.decisions[self.idx] = keep
        self._go(self.idx + 1 if self.idx < self.n - 1 else self.idx)

    @property
    def excluded(self):
        """Indices (0-based) marked for removal."""
        return [i for i, d in enumerate(self.decisions) if d is False]

    def result(self):
        """Series without the removed projections."""
        keep = [i for i, d in enumerate(self.decisions) if d is not False]
        if isinstance(self.series, TiltSeries):
            out = self.series.select(keep)
            out.meta["excluded_indices"] = self.excluded
            return out
        return self.data[keep]


def select_images_gui(series, display_size=384):
    """Manual review of the projections with Keep/Remove buttons
    (``select_images_gui_2`` without an exclusion list)."""
    return SelectImagesGUI(series, display_size)


# =========================================================================== tilt axis
def _rec_slice(stack, angles, row, algorithm, n_iter, pad_size=5):
    ts = TiltSeries(data=stack[:, row:row + 1, :], angles=angles)
    rec = reconstruct(ts, algorithm, int(n_iter), True, pad_size, print_time=False)[0]
    rec = rec - rec.min()
    mx = rec.max()
    return rec / mx if mx > 0 else rec


class TiltAxisGUI(_GUIBase):
    """Tilt axis shift/rotation with live slice reconstructions, see :func:`align_tilt_axis_gui`."""

    def __init__(self, series, algorithm="em", n_iter=15, display_size=420, rec_size=230):
        import ipywidgets as w
        if not isinstance(series, TiltSeries):
            raise TypeError("align_tilt_axis_gui needs a TiltSeries (data + angles)")
        self.series = series
        self.data = np.asarray(series.data, dtype=np.float32)
        self.current = self.data
        self.n_rows, self.n_cols = self.data.shape[-2:]
        self.contrast = 1.0
        rows = [int(round(self.n_rows / 4)) - 1, int(round(self.n_rows / 2)) - 1,
                int(round(3 * self.n_rows / 4)) - 1]
        rows = [int(np.clip(r, 0, self.n_rows - 1)) for r in rows]

        self.mip = _ImageCanvas((self.n_rows, self.n_cols), display_size)
        self.mip.set_image(self.data.max(axis=0))
        self.init_canvases = [_ImageCanvas((self.n_cols, self.n_cols), rec_size) for _ in range(3)]
        self.cur_canvases = [_ImageCanvas((self.n_cols, self.n_cols), rec_size) for _ in range(3)]
        colors = [_BLUE, _ORANGE, _GREEN]
        self.slice_sliders = [w.IntSlider(value=r, min=0, max=self.n_rows - 1,
                                          description=f"Slice {i + 1}", continuous_update=True,
                                          style={"handle_color": c}, layout=w.Layout(width="420px"))
                              for i, (r, c) in enumerate(zip(rows, colors))]
        self.rot_slider = w.FloatSlider(value=0, min=-90, max=90, step=0.1, description="Rotation",
                                        readout_format=".1f", continuous_update=True,
                                        layout=w.Layout(width="420px"))
        self.shift_slider = w.FloatSlider(value=0, min=-self.n_cols / 2, max=self.n_cols / 2, step=0.5,
                                          description="Translation", readout_format=".1f",
                                          continuous_update=True, layout=w.Layout(width="420px"))
        self.alg = w.ToggleButtons(options=["EM", "SIRT"], value=algorithm.upper(), description="Algorithm")
        self.n_iter = w.BoundedIntText(value=n_iter, min=1, max=10000, description="Iterations",
                                       layout=w.Layout(width="180px"))
        self.preview_btn = w.Button(description="Preview", button_style="primary", icon="refresh")
        self.cplus = w.Button(description="+", layout=w.Layout(width="40px"), tooltip="More contrast")
        self.cminus = w.Button(description="−", layout=w.Layout(width="40px"), tooltip="Less contrast")
        self.creset = w.Button(description="Reset", layout=w.Layout(width="70px"))
        self.status = w.HTML()
        for s in self.slice_sliders + [self.rot_slider, self.shift_slider]:
            s.observe(lambda c: self._draw_lines(), names="value")
        self.preview_btn.on_click(lambda b: self.preview())
        self.cplus.on_click(lambda b: self._set_contrast(self.contrast / 1.5))
        self.cminus.on_click(lambda b: self._set_contrast(self.contrast * 1.5))
        self.creset.on_click(lambda b: self._set_contrast(1.0))
        self.mip.mc.on_mouse_down(self._on_click)

        self._recs_init = [None] * 3
        self._recs_cur = [None] * 3
        self._labels_init = [w.HTML() for _ in range(3)]
        self._labels_cur = [w.HTML() for _ in range(3)]
        # one row per slice: initial (left) and preview (right)
        rec_grid = w.GridBox(
            [w.VBox([lab, c.mc]) for i in range(3)
             for lab, c in ((self._labels_init[i], self.init_canvases[i]),
                            (self._labels_cur[i], self.cur_canvases[i]))],
            layout=w.Layout(grid_template_columns=f"repeat(2, {rec_size + 10}px)"))
        controls = w.VBox(self.slice_sliders + [self.rot_slider, self.shift_slider,
                                                w.HBox([self.alg, self.n_iter]),
                                                w.HBox([self.preview_btn, w.Label("Contrast:"),
                                                        self.cminus, self.creset, self.cplus]),
                                                self.status,
                                                _help("Red: tilt axis. Positive translation moves the axis "
                                                      "to the right, positive rotation turns it clockwise. "
                                                      "Click on the projection to move slice 2. Left: "
                                                      "initial slices, right: preview. Apply with "
                                                      "<code>series = gui.result()</code>.")])
        # projection above the controls, slices on the right (they wrap below
        # only if the output area is too narrow)
        self.widget = w.HBox([w.VBox([self.mip.mc, controls]), rec_grid],
                             layout=w.Layout(flex_flow="row wrap", align_items="flex-start"))
        self._draw_lines()
        self._compute(initial=True)

    # ---------------------------------------------------------------- state
    @property
    def translation(self):
        """Tilt axis translation in pixels (slider steps rounded)."""
        return round(float(self.shift_slider.value), 3)

    @property
    def rotation(self):
        """Tilt axis rotation in degrees (slider steps rounded)."""
        return round(float(self.rot_slider.value), 3)

    def _draw_lines(self):
        from ipycanvas import hold_canvas
        rot = np.deg2rad(self.rotation)
        nr, nc = self.n_rows, self.n_cols
        with hold_canvas(self.mip.overlay):
            self.mip.clear_overlay()
            sh = np.sin(-rot) * nc / 2
            for s, c in zip(self.slice_sliders, [_BLUE, _ORANGE, _GREEN]):
                self.mip.line(0, s.value + sh, nc - 1, s.value - sh, c, 2)
            sh = np.tan(-rot) * nr / 2
            xc = (nc - 1) / 2 + self.translation
            self.mip.line(xc - sh, 0, xc + sh, nr - 1, "red", 2)

    def _on_click(self, x, y):
        _, yi = self.mip.to_image(x, y)
        self.slice_sliders[1].value = int(np.clip(round(yi - 0.5), 0, self.n_rows - 1))

    def _set_contrast(self, c):
        self.contrast = c
        for recs, canv in ((self._recs_init, self.init_canvases), (self._recs_cur, self.cur_canvases)):
            for r, cv in zip(recs, canv):
                if r is not None:
                    cv.set_image(r, 0, self.contrast)

    def _compute(self, initial=False):
        alg = self.alg.value.lower()
        n = self.n_iter.value
        for i, s in enumerate(self.slice_sliders):
            rec = _rec_slice(self.current, self.series.angles, s.value, alg, n)
            self._recs_cur[i] = rec
            self.cur_canvases[i].set_image(rec, 0, self.contrast)
            self._labels_cur[i].value = f"<b>Preview</b> slice {s.value}"
            if initial:
                self._recs_init[i] = rec
                self.init_canvases[i].set_image(rec, 0, self.contrast)
                self._labels_init[i].value = f"<b>Initial</b> slice {s.value}"

    def preview(self):
        """Apply the current tilt axis correction and reconstruct the 3 slices."""
        self.status.value = "<i>computing...</i>"
        try:
            self.current = apply_tilt_axis(self.data, self.translation, self.rotation)
            self._compute()
            self.status.value = (f"translation {self.translation:.1f} px, "
                                 f"rotation {self.rotation:.1f} deg")
        except Exception as e:   # keep the GUI alive
            self.status.value = f"<span style='color:red'>{e}</span>"

    def result(self):
        """The tilt series with the current translation/rotation applied."""
        return apply_tilt_axis(self.series, self.translation, self.rotation)


def align_tilt_axis_gui(series, algorithm="em", n_iter=15, display_size=420):
    """Interactive tilt axis alignment (``align_tilt_axis_gui``).

    Three slices are reconstructed (EM or SIRT); adjust the axis translation
    and rotation, press *Preview* to see the effect on the slices (tilt axis
    misalignment produces arcs/banana artefacts). Get the corrected series
    with ``gui.result()``."""
    return TiltAxisGUI(series, algorithm, n_iter, display_size)


# =========================================================================== manual alignment
class ManualAlignGUI(_GUIBase):
    """Colour overlay of successive images, shifted with buttons or the keyboard,
    see :func:`align_stack_manual_gui`."""

    def __init__(self, stack, reference=None, display_size=480):
        import ipywidgets as w
        self.input = stack
        self.data = np.array(_arr(stack), dtype=np.float32, copy=True)
        self.backup = self.data.copy()
        self.ref = None if reference is None else np.asarray(_arr(reference), dtype=np.float32)
        if self.ref is not None and self.ref.shape != self.data.shape:
            raise ValueError("Sizes of the target and the reference stacks have to be equal.")
        self.n = self.data.shape[0]
        self.shifts = np.zeros((self.n, 2), dtype=int)
        self.idx = 0
        self.canvas = _ImageCanvas(self.data.shape[-2:], display_size)
        b = dict(layout=w.Layout(width="44px"))
        self.up = w.Button(icon="arrow-up", **b)
        self.down = w.Button(icon="arrow-down", **b)
        self.left = w.Button(icon="arrow-left", **b)
        self.right = w.Button(icon="arrow-right", **b)
        self.step = w.BoundedIntText(value=1, min=1, max=1024, description="Step (px)",
                                     layout=w.Layout(width="160px"))
        self.mode = w.RadioButtons(options=["Single image", "All images"], value="Single image")
        self.prev_btn = w.Button(description="Previous", icon="backward")
        self.next_btn = w.Button(description="Next", icon="forward")
        last = self.n - 1 if self.ref is not None else max(0, self.n - 2)
        self.index = w.BoundedIntText(value=0, min=0, max=last, description="Image #",
                                      layout=w.Layout(width="160px"))
        self.reset_btn = w.Button(description="Reset all", icon="undo")
        self.info = w.HTML()
        for btn, d in ((self.up, "up"), (self.down, "down"), (self.left, "left"), (self.right, "right")):
            btn.on_click(lambda _b, d=d: self.shift(d))
        self.prev_btn.on_click(lambda _b: self._go(self.idx - 1))
        self.next_btn.on_click(lambda _b: self._go(self.idx + 1))
        self.index.observe(lambda c: self._go(c["new"]), names="value")
        self.reset_btn.on_click(lambda _b: self.reset())
        arrows = w.GridBox([w.Label(), self.up, w.Label(), self.left, self.down, self.right],
                           layout=w.Layout(grid_template_columns="repeat(3, 48px)"))
        self.view = w.Box([self.canvas.mc])
        self._keyboard()
        controls = w.VBox([w.HBox([self.prev_btn, self.index, self.next_btn]), arrows, self.step,
                           self.mode, self.reset_btn, self.info,
                           _help("Green: image <i>i</i>, magenta: image <i>i</i>+1 (shifted). "
                                 "Keyboard (mouse over the image): arrows shift, A/D previous/next, "
                                 "PageUp/PageDown double/halve the step. Get the aligned stack with "
                                 "<code>gui.result()</code>.")])
        self.widget = w.HBox([self.view, controls])
        self._draw()

    def _keyboard(self):
        try:
            from ipyevents import Event
        except ImportError:
            return
        self._events = Event(source=self.view, watched_events=["keydown"], prevent_default_action=True)
        self._events.on_dom_event(self._on_key)

    def _on_key(self, event):
        key = event.get("key", "")
        if key in ("ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"):
            self.shift(key[5:].lower())
        elif key in ("a", "A"):
            self._go(self.idx - 1)
        elif key in ("d", "D"):
            self._go(self.idx + 1)
        elif key == "PageUp":
            self.step.value = min(self.step.max, self.step.value * 2)
        elif key == "PageDown" and self.step.value > 1:
            self.step.value = max(1, self.step.value // 2)

    def _target(self):
        return self.idx if self.ref is not None else self.idx + 1

    def _go(self, idx):
        self.idx = int(np.clip(idx, self.index.min, self.index.max))
        if self.index.value != self.idx:
            self.index.value = self.idx
        self._draw()

    def _draw(self):
        a = self.data[self.idx]
        bimg = self.ref[self.idx] if self.ref is not None else self.data[min(self.idx + 1, self.n - 1)]

        def sc(x):
            lo, hi = float(x.min()), float(x.max())
            return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
        A, B = sc(a), sc(bimg)
        rgb = (np.stack([B, A, B], axis=-1) * 255).astype(np.uint8)   # A green, B magenta (imshowpair)
        self.canvas.set_image(rgb)
        t = self._target()
        self.info.value = (f"Pair {self.idx} / {self.index.max}: shift of image {t}: "
                           f"{self.shifts[t, 0]:+d} rows, {self.shifts[t, 1]:+d} cols")

    def shift(self, direction):
        """Shift by one step: 'up', 'down', 'left' or 'right' (circular shift
        like MATLAB ``circshift``)."""
        s = int(self.step.value)
        d = {"up": (-s, 0), "down": (s, 0), "left": (0, -s), "right": (0, s)}[direction]
        if self.mode.value == "Single image":
            t = self._target()
            self.data[t] = np.roll(self.data[t], d, axis=(0, 1))
            self.shifts[t] += d
        else:
            self.data = np.roll(self.data, d, axis=(1, 2))
            self.shifts += np.array(d)
        self._draw()

    def reset(self):
        self.data = self.backup.copy()
        self.shifts[:] = 0
        self._draw()

    def result(self):
        """Aligned stack (same type as the input); the applied [row, col]
        displacements are in ``meta['manual_shifts']`` for a TiltSeries."""
        if isinstance(self.input, TiltSeries):
            out = self.input.with_data(self.data.copy())
            out.meta["manual_shifts"] = self.shifts.copy()
            return out
        return self.data.copy()


def align_stack_manual_gui(stack, reference=None, display_size=480):
    """Manual alignment of successive images (``align_stack_manual``).

    Pairs of successive images are shown as a green/magenta overlay; shift the
    second image with the arrow buttons or keys (or all images at once).
    With ``reference`` (same shape), each image is compared to its reference
    instead. Get the aligned stack with ``gui.result()``."""
    return ManualAlignGUI(stack, reference, display_size)


# =========================================================================== cupping correction
class CuppingCorrectionGUI(_GUIBase):
    """Cupping correction strength with a slice reconstruction preview, see
    :func:`cupping_correction_gui`."""

    def __init__(self, series, algorithm="em", n_iter=15, display_size=380, rec_size=380):
        import ipywidgets as w
        if not isinstance(series, TiltSeries):
            raise TypeError("cupping_correction_gui needs a TiltSeries (data + angles)")
        self.series = series
        self.data = np.asarray(series.data, dtype=np.float32)
        self.n_rows, self.n_cols = self.data.shape[-2:]
        self.full_range = intensity_range(self.data, "full")
        self.mip = _ImageCanvas((self.n_rows, self.n_cols), display_size)
        self.mip.set_image(self.data.max(axis=0))
        self.rec_canvas = _ImageCanvas((self.n_cols, self.n_cols), rec_size)
        self.strength = w.FloatSlider(value=0.0, min=0.0, max=4.0, step=0.1, description="Strength",
                                      readout_format=".1f", continuous_update=False,
                                      layout=w.Layout(width="360px"))
        self.range_mode = w.ToggleButtons(options=["Full", "Robust", "Manual"], value="Full",
                                          description="Range")
        self.i_min = w.FloatText(value=round(self.full_range[0], 6), description="min", disabled=True,
                                 layout=w.Layout(width="180px"))
        self.i_max = w.FloatText(value=round(self.full_range[1], 6), description="max", disabled=True,
                                 layout=w.Layout(width="180px"))
        self.slice = w.IntSlider(value=int(round(self.n_rows / 2)) - 1, min=0, max=self.n_rows - 1,
                                 description="Slice", continuous_update=True, layout=w.Layout(width="360px"))
        self.alg = w.ToggleButtons(options=["EM", "SIRT"], value=algorithm.upper(), description="Algorithm")
        self.n_iter = w.BoundedIntText(value=n_iter, min=1, max=10000, description="Iterations",
                                       layout=w.Layout(width="180px"))
        self.preview_btn = w.Button(description="Preview", button_style="primary", icon="refresh")
        self.status = w.HTML()
        self.range_mode.observe(self._on_range, names="value")
        self.slice.observe(lambda c: self._draw_line(), names="value")
        self.preview_btn.on_click(lambda b: self.preview())
        self.mip.mc.on_mouse_down(self._on_click)
        controls = w.VBox([self.strength, self.range_mode, w.HBox([self.i_min, self.i_max]), self.slice,
                           w.HBox([self.alg, self.n_iter]), self.preview_btn, self.status,
                           _help("Increase the strength until the reconstructed slice has no darker "
                                 "core (cupping). Click on the projection to select the slice. Apply with "
                                 "<code>series = gui.result()</code>.")])
        self.widget = w.VBox([w.HBox([self.mip.mc, self.rec_canvas.mc]), controls])
        self._draw_line()
        self.preview()

    def _on_range(self, change):
        mode = change["new"]
        manual = mode == "Manual"
        self.i_min.disabled = self.i_max.disabled = not manual
        if not manual:
            lo, hi = intensity_range(self.data, mode.lower())
            self.i_min.value, self.i_max.value = round(lo, 6), round(hi, 6)

    def _on_click(self, x, y):
        _, yi = self.mip.to_image(x, y)
        self.slice.value = int(np.clip(round(yi - 0.5), 0, self.n_rows - 1))

    def _draw_line(self):
        from ipycanvas import hold_canvas
        with hold_canvas(self.mip.overlay):
            self.mip.clear_overlay()
            self.mip.line(0, self.slice.value, self.n_cols - 1, self.slice.value, "red", 2)

    @property
    def params(self):
        """Current correction parameters (strength, i_min, i_max)."""
        return dict(strength=round(float(self.strength.value), 3), i_min=float(self.i_min.value),
                    i_max=float(self.i_max.value))

    def preview(self):
        """Reconstruct the selected slice from the corrected projections."""
        self.status.value = "<i>computing...</i>"
        try:
            corr = cupping_correction(self.data, **self.params)
            rec = reconstruct(TiltSeries(data=corr[:, self.slice.value:self.slice.value + 1, :],
                                         angles=self.series.angles),
                              self.alg.value.lower(), int(self.n_iter.value), True, 20,
                              print_time=False)[0]
            self.rec_canvas.set_image(rec)
            p = self.params
            self.status.value = (f"slice {self.slice.value}: strength {p['strength']:.1f}, "
                                 f"range [{p['i_min']:.4g}, {p['i_max']:.4g}]")
        except Exception as e:
            self.status.value = f"<span style='color:red'>{e}</span>"

    def result(self):
        """The corrected tilt series."""
        return cupping_correction(self.series, **self.params)


def cupping_correction_gui(series, algorithm="em", n_iter=15):
    """Interactive cupping correction (``cupping_correction_gui``): adjust the
    correction strength while previewing a reconstructed slice. Get the
    corrected series with ``gui.result()``."""
    return CuppingCorrectionGUI(series, algorithm, n_iter)
