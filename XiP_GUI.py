"""
XiP GUI — X-ray Interactive Plotter
PyQt6 front-end for XCMDataPlotter (XiP_v2.py).
"""

import os
import sys
import traceback
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

# Set the Qt backend BEFORE any pyplot import so the embedded canvas works.
import matplotlib
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar


class _HighDpiToolbar(NavigationToolbar):
    """Navigation toolbar that saves figures at 300 dpi."""

    def save_figure(self, *args):
        orig_dpi = self.canvas.figure.dpi
        try:
            self.canvas.figure.set_dpi(300)
            super().save_figure(*args)
        finally:
            self.canvas.figure.set_dpi(orig_dpi)
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QFont

from xip.backend import XCMDataPlotter
from xip.plotting import KIND_STYLES, render_xspec_plot
from xip.xcm_editor import XCMParamEditorWidget, validate_xcm_text
from xip.profiles import load_profiles
from xip.profiles.base import XiPProfile
from xspec import Fit, Xset  # xspec already loaded by backend

# Discover profiles once at startup.
_PROFILES: list[XiPProfile] = load_profiles()
_PROFILES_BY_NAME: dict[str, XiPProfile] = {p.name: p for p in _PROFILES}


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class PlotWorker(QThread):
    """Runs the XSPEC session restore + fit in a background thread."""

    finished = pyqtSignal(dict, dict, dict, object, object)
    error_occurred = pyqtSignal(str)
    log_message = pyqtSignal(str)

    def __init__(self, xcm_file, energy_range, min_sig, max_bins, perform_fit):
        super().__init__()
        self.xcm_file = xcm_file
        self.energy_range = energy_range
        self.min_sig = min_sig
        self.max_bins = max_bins
        self.perform_fit = perform_fit

    def run(self):
        try:
            plotter = XCMDataPlotter(
                xcm_file=self.xcm_file,
                energy_range=self.energy_range,
                min_sig=self.min_sig,
                max_bins=self.max_bins,
                perform_fit=self.perform_fit,
            )
            self.log_message.emit(f"Loading: {self.xcm_file}\n")
            plotter._configure_xspec()
            self.log_message.emit("Restoring XSPEC session and Performing fit……\n")
            plotter._restore_session()
            
            if self.perform_fit and plotter.statistic is not None and plotter.dof:
                stat, dof = plotter.statistic, plotter.dof
                self.log_message.emit(
                    f"CSTAT/dof = {stat/dof:.3f}  ({stat:.2f} / {dof})\n"
                )
            spectra, residuals, component_curves = plotter._collect_plot_arrays()
            self.log_message.emit("Done.\n")
            self._plotter = plotter  # store for main-thread slider access
            self.finished.emit(
                spectra, residuals, component_curves,
                plotter.statistic, plotter.dof,
            )
        except Exception as exc:
            self._plotter = None
            self.error_occurred.emit(f"{exc}\n\n{traceback.format_exc()}")

    @property
    def plotter(self):
        return getattr(self, "_plotter", None)


# ---------------------------------------------------------------------------
# Matplotlib canvas
# ---------------------------------------------------------------------------


class XSpecCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(9, 6), dpi=100, tight_layout=False)
        gs = GridSpec(2, 1, figure=self.fig, height_ratios=[3, 1], hspace=0)
        self.ax_top = self.fig.add_subplot(gs[0])
        self.ax_bot = self.fig.add_subplot(gs[1], sharex=self.ax_top)
        self._draw_placeholder()

        super().__init__(self.fig)
        if parent:
            self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()

    # ------------------------------------------------------------------
    def _draw_placeholder(self):
        self.ax_top.set_visible(False)
        self.ax_bot.set_visible(False)
        self.fig.text(
            0.5, 0.5,
            "Select an XCM file and click  Plot",
            ha="center", va="center",
            fontsize=14, color="gray",
            transform=self.fig.transFigure,
        )

    def _clear_placeholder(self):
        self.fig.texts.clear()
        self.ax_top.set_visible(True)
        self.ax_bot.set_visible(True)

    # ------------------------------------------------------------------
    def plot_spectra(self, spectra, residuals, component_curves, statistic, dof,
                     modes=None, profile="eROSITA", energy_range=None):
        """
        profile="eROSITA"
          modes is a set of ints (any combination):
            1 — Source spectrum: data + source model + source components
            2 — Source spectrum: data + source model + bkg components on source
            3 — Background spectrum: data + bkg model + bkg components on bkg
        profile="Default"
          modes:
            1 — Source spectrum: data + source model + all components
            3 — Background spectrum: data + bkg model + all bkg components
        """
        if not modes:
            modes = {1}
        self._clear_placeholder()
        render_xspec_plot(
            self.ax_top, self.ax_bot,
            spectra, residuals, component_curves,
            statistic, dof,
            modes=modes,
            profile=profile,
            energy_range=energy_range,
        )
        self.fig.tight_layout(rect=[0, 0, 1, 1])
        self.draw()


# ---------------------------------------------------------------------------
# Parameter slider tab
# ---------------------------------------------------------------------------

_N_SLIDER_STEPS = 1000


class _ParamSliderRow(QWidget):
    """One row: component.param  |  slider  |  current-value label  |  [range]."""

    value_changed = pyqtSignal(str, str, str, float)  # model_name, comp_name, param_name, value

    def __init__(
        self,
        model_name: str,
        comp_name: str,
        param_name: str,
        value: float,
        vmin: float,
        vmax: float,
        use_log: bool,
        parent=None,
    ):
        super().__init__(parent)
        self._model_name = model_name
        self._comp_name = comp_name
        self._param_name = param_name
        self._vmin = vmin
        self._vmax = vmax
        self._use_log = use_log

        # Debounce timer — fires 80 ms after the last slider movement
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._emit_change)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(6)

        lbl = QLabel(f"{comp_name}.{param_name}")
        lbl.setFixedWidth(160)
        lbl.setFont(QFont("Courier New", 9))
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(lbl)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, _N_SLIDER_STEPS)
        self._slider.setValue(self._value_to_pos(value))
        self._slider.valueChanged.connect(self._on_slider_changed)
        lay.addWidget(self._slider, stretch=1)

        self._val_lbl = QLabel(self._fmt(value))
        self._val_lbl.setFixedWidth(80)
        self._val_lbl.setFont(QFont("Courier New", 9))
        self._val_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(self._val_lbl)

        range_lbl = QLabel(f"[{self._fmt(vmin)}, {self._fmt(vmax)}]")
        range_lbl.setStyleSheet("color: #888; font-size: 9px;")
        range_lbl.setFixedWidth(140)
        lay.addWidget(range_lbl)

        self._orig_value = value  # stored for Reset

    # ------------------------------------------------------------------
    @staticmethod
    def _fmt(v: float) -> str:
        if v == 0:
            return "0"
        a = abs(v)
        if 0.01 <= a < 1e4:
            return f"{v:.4g}"
        return f"{v:.3e}"

    def _value_to_pos(self, value: float) -> int:
        v = max(self._vmin, min(self._vmax, value))
        if self._use_log:
            lo, hi = np.log10(self._vmin), np.log10(self._vmax)
            pos = (np.log10(v) - lo) / (hi - lo) * _N_SLIDER_STEPS
        else:
            pos = (v - self._vmin) / (self._vmax - self._vmin) * _N_SLIDER_STEPS
        return int(np.clip(pos, 0, _N_SLIDER_STEPS))

    def _pos_to_value(self, pos: int) -> float:
        if self._use_log:
            lo, hi = np.log10(self._vmin), np.log10(self._vmax)
            return 10 ** (lo + pos / _N_SLIDER_STEPS * (hi - lo))
        return self._vmin + pos / _N_SLIDER_STEPS * (self._vmax - self._vmin)

    # ------------------------------------------------------------------
    def _on_slider_changed(self, pos: int):
        self._val_lbl.setText(self._fmt(self._pos_to_value(pos)))
        self._timer.start()  # (re-)start debounce

    def _emit_change(self):
        v = self._pos_to_value(self._slider.value())
        self.value_changed.emit(self._model_name, self._comp_name, self._param_name, v)

    def set_value(self, value: float):
        """Programmatically move the slider without emitting a change signal."""
        self._slider.blockSignals(True)
        self._slider.setValue(self._value_to_pos(value))
        self._val_lbl.setText(self._fmt(value))
        self._slider.blockSignals(False)

    def reset(self):
        """Silently restore the slider to its original value (no signal emitted)."""
        self.set_value(self._orig_value)


class ParamSlidersWidget(QWidget):
    """Scrollable panel of sliders for all non-frozen XSPEC parameters."""

    param_changed = pyqtSignal(str, str, str, float)  # (model_name, comp_name, param_name, value)
    reset_requested = pyqtSignal(list)  # list of (model_name, comp_name, param_name, orig_value)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[_ParamSliderRow] = []
        self._orig_data: list[tuple] = []  # (model_name, comp_name, param_name, orig_value)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header row with Reset button
        header = QWidget()
        header_lay = QHBoxLayout(header)
        header_lay.setContentsMargins(6, 4, 6, 4)
        param_hdr_lbl = QLabel("Parameters")
        param_hdr_f = QFont()
        param_hdr_f.setBold(True)
        param_hdr_lbl.setFont(param_hdr_f)
        header_lay.addWidget(param_hdr_lbl)
        header_lay.addStretch()
        self._reset_btn = QPushButton("Reset to fit values")
        self._reset_btn.setFixedHeight(26)
        self._reset_btn.setEnabled(False)
        self._reset_btn.clicked.connect(self._on_reset_clicked)
        header_lay.addWidget(self._reset_btn)
        outer.addWidget(header)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        outer.addWidget(self._scroll)

        self._content = QWidget()
        self._content_lay = QVBoxLayout(self._content)
        self._content_lay.setContentsMargins(8, 8, 8, 8)
        self._content_lay.setSpacing(8)

        self._placeholder = QLabel(
            "Load and plot an XCM file to see parameter sliders here."
        )
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: gray; font-size: 13px;")
        self._content_lay.addWidget(self._placeholder)
        self._content_lay.addStretch()

        self._scroll.setWidget(self._content)

    # ------------------------------------------------------------------
    def clear(self):
        """Remove all slider rows and group boxes, show placeholder."""
        for row in self._rows:
            row.setParent(None)
        self._rows.clear()
        self._orig_data.clear()
        while self._content_lay.count() > 0:
            item = self._content_lay.takeAt(0)
            w = item.widget() if item else None
            if w and w is not self._placeholder:
                w.setParent(None)
        self._placeholder.show()
        self._content_lay.addWidget(self._placeholder)
        self._content_lay.addStretch()

    # ------------------------------------------------------------------
    def build(self, plotter) -> None:
        """Populate sliders from the current XSPEC AllModels state."""
        self.clear()
        # Remove trailing stretch so we can append groups cleanly
        last_idx = self._content_lay.count() - 1
        if last_idx >= 0:
            item = self._content_lay.takeAt(last_idx)
            if item and not item.widget():
                del item  # discard the stretch spacer

        self._placeholder.hide()
        found_any = False

        for model_name in plotter.model_order:
            model_obj = plotter._get_model_object(model_name)
            if model_obj is None:
                continue

            rows_for_model: list[_ParamSliderRow] = []
            for comp_name in model_obj.componentNames:
                comp = getattr(model_obj, comp_name, None)
                if comp is None:
                    continue
                for param_name in comp.parameterNames:
                    param = getattr(comp, param_name, None)
                    if param is None or param.frozen:
                        continue
                    vals = param.values  # (value, delta, hard_min, soft_min, soft_max, hard_max)
                    value, hard_min, soft_max = vals[0], vals[2], vals[4]
                    if value == 0:
                        # Zero-value fallback: use hard bounds or symmetric ±1 range
                        vmin = hard_min if hard_min < 0 else -1.0
                        vmax = soft_max if 0 < soft_max < 1e5 else 1.0
                    else:
                        # XiP.py-style limits: val ± 3*|val|, clipped to hard bounds
                        vmin = max(hard_min, value - abs(value * 3))
                        vmax = min(soft_max, value + abs(value * 3))
                    # Match utils.py convention: linear for PhoIndex / nH / factor,
                    # log for everything else (guard: both bounds must be positive)
                    _lin_params = ("PhoIndex", "nH", "factor")
                    use_log = (
                        not any(lp in param_name for lp in _lin_params)
                        and vmin > 0 and vmax > 0
                    )
                    # For log scale ensure vmin is strictly positive
                    if use_log and vmin <= 0:
                        vmin = max(hard_min if hard_min > 0 else value * 1e-3, value * 1e-3)
                    if vmin >= vmax:
                        continue
                    row = _ParamSliderRow(
                        model_name, comp_name, param_name,
                        value, vmin, vmax, use_log,
                        self._content,
                    )
                    row.value_changed.connect(self.param_changed)
                    rows_for_model.append(row)
                    self._rows.append(row)
                    self._orig_data.append((model_name, comp_name, param_name, value))

            if rows_for_model:
                found_any = True
                kind = plotter.model_kind_map.get(model_name, "other")
                grp = QGroupBox(
                    f"{model_name}  ({kind.replace('_', ' ').title()})"
                )
                grp_f = QFont()
                grp_f.setBold(True)
                grp.setFont(grp_f)
                grp_lay = QVBoxLayout(grp)
                grp_lay.setSpacing(2)
                grp_lay.setContentsMargins(6, 10, 6, 6)
                for row in rows_for_model:
                    grp_lay.addWidget(row)
                self._content_lay.addWidget(grp)

        if not found_any:
            self._placeholder.setText(
                "No free (non-frozen) parameters found in the loaded model."
            )
            self._placeholder.show()
            self._content_lay.addWidget(self._placeholder)
        else:
            self._reset_btn.setEnabled(True)

        self._content_lay.addStretch()

    def _on_reset_clicked(self):
        """Silently restore all sliders then notify the main window to replot."""
        for row in self._rows:
            row.reset()
        self.reset_requested.emit(list(self._orig_data))

    # ------------------------------------------------------------------
    def setEnabled(self, enabled: bool) -> None:
        for row in self._rows:
            row._slider.setEnabled(enabled)
        super().setEnabled(enabled)


# ---------------------------------------------------------------------------
# Fast replot worker  (slider / reset interactions)
# ---------------------------------------------------------------------------


class ReplotWorker(QThread):
    """Applies parameter changes in-place and re-collects plot arrays."""

    finished = pyqtSignal(dict, dict, dict, object, object)
    error_occurred = pyqtSignal(str)

    def __init__(self, plotter, changes: list):
        super().__init__()
        self._plotter = plotter
        self._changes = changes  # [(model_name, comp_name, param_name, value), ...]

    def run(self):
        try:
            for model_name, comp_name, param_name, value in self._changes:
                model_obj = self._plotter._get_model_object(model_name)
                if model_obj is None:
                    continue
                comp = getattr(model_obj, comp_name, None)
                if comp is None:
                    continue
                setattr(comp, param_name, value)
            spectra, residuals_by_spec, component_curves = \
                self._plotter._collect_plot_arrays(recompute_structure=False)
            try:
                stat = Fit.statistic
            except Exception:
                stat = self._plotter.statistic
            self.finished.emit(
                spectra, residuals_by_spec, component_curves,
                stat, self._plotter.dof,
            )
        except Exception as exc:
            self.error_occurred.emit(f"{exc}\n\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XiP — X-ray Interactive Plotter")
        self.resize(1280, 960)
        self._worker: PlotWorker | None = None
        self._replot_worker: ReplotWorker | None = None
        self._pending_param_changes: list | None = None
        self._last_plot_data: tuple | None = None  # (spectra, residuals, component_curves, stat, dof)
        self._plotter = None  # live XCMDataPlotter kept for slider updates
        self._setup_ui()

    # ------------------------------------------------------------------
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        root.addWidget(self._build_sidebar(), stretch=0)
        root.addWidget(self._build_right_panel(), stretch=1)

        self.statusBar().showMessage("Ready — select an XCM file and click Plot.")

    # ------------------------------------------------------------------
    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setFixedWidth(240)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Title
        title = QLabel("XiP")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        f = QFont()
        f.setPointSize(22)
        f.setBold(True)
        title.setFont(f)

        sub = QLabel("X-ray Interactive Plotter")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fs = QFont()
        fs.setPointSize(9)
        sub.setFont(fs)
        sub.setStyleSheet("color: gray;")

        layout.addWidget(title)
        layout.addWidget(sub)

        # --- File group ---
        file_grp = QGroupBox("Input File")
        file_lay = QVBoxLayout(file_grp)

        row = QHBoxLayout()
        self.xcm_edit = QLineEdit()
        self.xcm_edit.setPlaceholderText("Path to .xcm file…")
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(72)
        browse_btn.clicked.connect(self._browse_xcm)
        row.addWidget(self.xcm_edit)
        row.addWidget(browse_btn)
        file_lay.addLayout(row)

        self.profile_combo = QComboBox()
        for p in _PROFILES:
            self.profile_combo.addItem(p.name)
        self.profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        file_lay.addWidget(QLabel("Select model profile:"))
        file_lay.addWidget(self.profile_combo)

        layout.addWidget(file_grp)

        # --- Settings group ---
        settings_grp = QGroupBox("Plot Settings")
        form = QFormLayout(settings_grp)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        form.setVerticalSpacing(6)

        # Energy range
        erow = QHBoxLayout()
        self.emin_spin = QDoubleSpinBox()
        self.emin_spin.setRange(0.01, 100.0)
        self.emin_spin.setSingleStep(0.1)
        self.emin_spin.setValue(0.2)
        self.emin_spin.setDecimals(2)
        self.emin_spin.setSuffix(" keV")
        self.emax_spin = QDoubleSpinBox()
        self.emax_spin.setRange(0.01, 100.0)
        self.emax_spin.setSingleStep(0.5)
        self.emax_spin.setValue(8.0)
        self.emax_spin.setDecimals(2)
        self.emax_spin.setSuffix(" keV")
        erow.addWidget(QLabel("Min:"))
        erow.addWidget(self.emin_spin)
        erow.addWidget(QLabel("Max:"))
        erow.addWidget(self.emax_spin)
        form.addRow("Energy Range:", erow)

        self.no_filter_cb = QCheckBox("Disable energy filter")
        self.no_filter_cb.stateChanged.connect(self._toggle_energy_filter)
        form.addRow("", self.no_filter_cb)

        self.min_sig_spin = QSpinBox()
        self.min_sig_spin.setRange(1, 200)
        self.min_sig_spin.setValue(12)
        form.addRow("Min Sigma:", self.min_sig_spin)

        self.max_bins_spin = QSpinBox()
        self.max_bins_spin.setRange(1, 2000)
        self.max_bins_spin.setValue(100)
        form.addRow("Max Bins:", self.max_bins_spin)

        self.fit_cb = QCheckBox("Perform fit after loading")
        self.fit_cb.setChecked(True)
        form.addRow("", self.fit_cb)

        layout.addWidget(settings_grp)

        # --- View mode ---
        self._vis_grp = QGroupBox("View Mode")
        self._vis_lay = QVBoxLayout(self._vis_grp)
        # _mode_checkboxes: list of (mode_number, QCheckBox)
        self._mode_checkboxes: list[tuple[int, QCheckBox]] = []
        # Build initial checkboxes for the first profile in the list
        initial_profile = _PROFILES[0] if _PROFILES else None
        self._rebuild_mode_checkboxes(initial_profile)
        layout.addWidget(self._vis_grp)

        # --- Plot button ---
        self.plot_btn = QPushButton("Plot")
        self.plot_btn.setFixedHeight(38)
        bf = QFont()
        bf.setPointSize(12)
        bf.setBold(True)
        self.plot_btn.setFont(bf)
        self.plot_btn.clicked.connect(self._run_plot)
        layout.addWidget(self.plot_btn)

        self.capture_btn = QPushButton("Save new params to XCM")
        self.capture_btn.setFixedHeight(30)
        self.capture_btn.setEnabled(False)
        self.capture_btn.clicked.connect(self._capture_params_to_xcm)
        layout.addWidget(self.capture_btn)

        # --- Log area ---
        log_grp = QGroupBox("Log")
        log_lay = QVBoxLayout(log_grp)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFixedHeight(180)
        self.log_view.setFont(QFont("Courier New", 9))
        log_lay.addWidget(self.log_view)
        layout.addWidget(log_grp)

        layout.addStretch()
        return sidebar

    # ------------------------------------------------------------------
    def _build_right_panel(self) -> QWidget:
        self.tabs = QTabWidget()

        # --- Plot tab (canvas on top, sliders below) ---
        plot_widget = QWidget()
        plot_outer = QVBoxLayout(plot_widget)
        plot_outer.setContentsMargins(0, 0, 0, 0)
        plot_outer.setSpacing(0)

        self.canvas = XSpecCanvas()
        self.toolbar = _HighDpiToolbar(self.canvas, self)
        plot_outer.addWidget(self.toolbar)

        # Vertical splitter: canvas on top, parameter sliders on bottom
        self.plot_splitter = QSplitter(Qt.Orientation.Vertical)
        self.plot_splitter.addWidget(self.canvas)

        self.sliders_tab = ParamSlidersWidget()
        self.sliders_tab.param_changed.connect(self._on_param_changed)
        self.sliders_tab.reset_requested.connect(self._on_reset_params)
        self.sliders_tab.setMinimumHeight(200)
        self.plot_splitter.addWidget(self.sliders_tab)
        self.plot_splitter.setStretchFactor(0, 1)
        self.plot_splitter.setStretchFactor(1, 1)

        plot_outer.addWidget(self.plot_splitter)
        self.tabs.addTab(plot_widget, "Plot")

        # --- XCM Editor tab ---
        editor_widget = QWidget()
        editor_lay = QVBoxLayout(editor_widget)
        editor_lay.setContentsMargins(6, 6, 6, 6)
        editor_lay.setSpacing(6)

        # Mode toggle row
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Editor mode:"))
        self.editor_mode_combo = QComboBox()
        self.editor_mode_combo.addItems(["Simple Text", "Structured"])
        self.editor_mode_combo.currentIndexChanged.connect(self._on_editor_mode_changed)
        mode_row.addWidget(self.editor_mode_combo)
        mode_row.addStretch()
        editor_lay.addLayout(mode_row)

        # Stacked widget: page 0 = plain text, page 1 = structured
        self.editor_stack = QStackedWidget()

        self.xcm_editor = QPlainTextEdit()
        self.xcm_editor.setFont(QFont("Courier New", 10))
        self.xcm_editor.setPlaceholderText("Load an XCM file to edit its contents here…")
        self.editor_stack.addWidget(self.xcm_editor)

        self.xcm_struct_editor = XCMParamEditorWidget()
        self.editor_stack.addWidget(self.xcm_struct_editor)

        editor_lay.addWidget(self.editor_stack)

        btn_row = QHBoxLayout()
        reload_btn = QPushButton("Reload from disk")
        reload_btn.clicked.connect(self._load_xcm_into_editor)
        self.refit_btn = QPushButton("Save & Refit")
        self.refit_btn.clicked.connect(self._save_and_refit)
        bf = QFont()
        bf.setBold(True)
        self.refit_btn.setFont(bf)
        btn_row.addWidget(reload_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.refit_btn)
        editor_lay.addLayout(btn_row)

        self.tabs.addTab(editor_widget, "XCM Editor")

        return self.tabs

    # ------------------------------------------------------------------
    def _toggle_energy_filter(self):
        enabled = not self.no_filter_cb.isChecked()
        self.emin_spin.setEnabled(enabled)
        self.emax_spin.setEnabled(enabled)

    def _rebuild_mode_checkboxes(self, profile: "XiPProfile | None") -> None:
        """Clear and rebuild the View Mode checkboxes for *profile*."""
        # Remove old widgets
        for _, cb in self._mode_checkboxes:
            self._vis_lay.removeWidget(cb)
            cb.deleteLater()
        self._mode_checkboxes = []

        if profile is None:
            return

        default_modes = profile.default_modes
        for mode_num, label in profile.modes:
            cb = QCheckBox(label)
            cb.setChecked(mode_num in default_modes)
            cb.stateChanged.connect(self._replot)
            self._vis_lay.addWidget(cb)
            self._mode_checkboxes.append((mode_num, cb))

    def _get_selected_profile(self) -> "XiPProfile | None":
        return _PROFILES_BY_NAME.get(self.profile_combo.currentText())

    def _on_profile_changed(self):
        profile = self._get_selected_profile()
        self._rebuild_mode_checkboxes(profile)
        self._replot()

    def _browse_xcm(self):
        current = self.xcm_edit.text().strip()
        if current and Path(current).exists():
            start_dir = str(Path(current).parent)
        else:
            start_dir = str(Path.cwd())
        path, _ = QFileDialog.getOpenFileName(
            self, "Open XCM File",
            start_dir,
            "XCM Files (*.xcm);;All Files (*)",
        )
        if path:
            self.xcm_edit.setText(path)
            self._load_xcm_into_editor()

    def _on_editor_mode_changed(self, index: int) -> None:
        if index == 1:  # switching to Structured
            self.xcm_struct_editor.load(self.xcm_editor.toPlainText())
        else:  # switching to Simple Text
            text = self.xcm_struct_editor.get_xcm_text()
            if text.strip():
                self.xcm_editor.setPlainText(text)
        self.editor_stack.setCurrentIndex(index)

    def _load_xcm_into_editor(self):
        path = self.xcm_edit.text().strip()
        if path and Path(path).exists():
            try:
                text = Path(path).read_text()
                self.xcm_editor.setPlainText(text)
                if self.editor_mode_combo.currentIndex() == 1:
                    self.xcm_struct_editor.load(text)
            except Exception as exc:
                self.xcm_editor.setPlainText(f"# Could not read file:\n# {exc}")

    def _save_and_refit(self):
        path = self.xcm_edit.text().strip()
        if not path:
            self.statusBar().showMessage("No XCM file path set.")
            return
        # Get text from whichever editor is active
        if self.editor_mode_combo.currentIndex() == 1:
            errors = self.xcm_struct_editor.validate_expressions()
            text = self.xcm_struct_editor.get_xcm_text()
            self.xcm_editor.setPlainText(text)
        else:
            text = self.xcm_editor.toPlainText()
            errors = validate_xcm_text(text)

        # Hard errors (✗) block the save; warnings (⚠) ask for confirmation
        hard_errors = [e for e in errors if e.startswith("✗")]
        warnings_only = [e for e in errors if e.startswith("⚠")]

        if hard_errors:
            QMessageBox.warning(
                self, "XCM Validation Error",
                "Cannot save — fix the following errors first:\n\n"
                + "\n\n".join(hard_errors),
            )
            return

        if warnings_only:
            reply = QMessageBox.question(
                self, "XCM Validation Warning",
                "The following components are not in the local XSPEC database:\n\n"
                + "\n\n".join(warnings_only)
                + "\n\nThey may still work if XSPEC knows them.  Continue saving?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        try:
            Path(path).write_text(text)
        except Exception as exc:
            self.statusBar().showMessage(f"Save failed: {exc}")
            return
        self.statusBar().showMessage(f"Saved {path} — refitting…")
        self.tabs.setCurrentIndex(0)
        self._run_plot()

    # ------------------------------------------------------------------
    def _run_plot(self):
        xcm_file = self.xcm_edit.text().strip()
        if not xcm_file:
            self.statusBar().showMessage("Please select an XCM file first.")
            return
        if not Path(xcm_file).exists():
            self.statusBar().showMessage(f"File not found: {xcm_file}")
            return
        if (self._worker and self._worker.isRunning()) or \
                (self._replot_worker and self._replot_worker.isRunning()):
            self.statusBar().showMessage("Analysis already running — please wait.")
            return

        energy_range = (
            None if self.no_filter_cb.isChecked()
            else (self.emin_spin.value(), self.emax_spin.value())
        )

        self.plot_btn.setEnabled(False)
        self.plot_btn.setText("Running…")
        self.sliders_tab.setEnabled(False)
        self.log_view.clear()
        self.statusBar().showMessage("Running XSPEC analysis…")

        self._worker = PlotWorker(
            xcm_file=xcm_file,
            energy_range=energy_range,
            min_sig=self.min_sig_spin.value(),
            max_bins=self.max_bins_spin.value(),
            perform_fit=self.fit_cb.isChecked(),
        )
        self._worker.finished.connect(self._on_finished)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.log_message.connect(self._append_log)
        self._worker.start()

    def _append_log(self, msg: str):
        self.log_view.append(msg)

    def _replot(self):
        if self._last_plot_data is None:
            return
        spectra, residuals, component_curves, statistic, dof = self._last_plot_data
        modes = {num for num, cb in self._mode_checkboxes if cb.isChecked()}
        energy_range = (
            None if self.no_filter_cb.isChecked()
            else (self.emin_spin.value(), self.emax_spin.value())
        )
        self.canvas.plot_spectra(
            spectra, residuals, component_curves, statistic, dof,
            modes=modes or {self._mode_checkboxes[0][0]} if self._mode_checkboxes else {1},
            profile=self._get_selected_profile(),
            energy_range=energy_range,
        )

    def _on_finished(self, spectra, residuals, component_curves, statistic, dof):
        self._last_plot_data = (spectra, residuals, component_curves, statistic, dof)
        self._plotter = self._worker.plotter
        if self._plotter is not None:
            self.sliders_tab.build(self._plotter)
            self.capture_btn.setEnabled(True)
            # Auto-select the first profile whose detect() returns True;
            # fall back to "Default" if none match.
            mkm = self._plotter.model_kind_map
            detected_name = "Default"
            for p in _PROFILES:
                if p.detect(mkm):
                    detected_name = p.name
                    break
            # Block the combo signal so _on_profile_changed doesn't fire twice.
            self.profile_combo.blockSignals(True)
            self.profile_combo.setCurrentText(detected_name)
            self.profile_combo.blockSignals(False)
            # Rebuild mode checkboxes for the auto-selected profile.
            self._rebuild_mode_checkboxes(_PROFILES_BY_NAME.get(detected_name))
        self._replot()
        self.plot_btn.setEnabled(True)
        self.plot_btn.setText("Plot")
        self.sliders_tab.setEnabled(True)
        self.statusBar().showMessage("Plot complete.")

    def _on_param_changed(self, model_name: str, comp_name: str, param_name: str, value: float):
        """Queue a parameter change and dispatch an asynchronous replot."""
        if self._plotter is None:
            return
        self._pending_param_changes = [(model_name, comp_name, param_name, value)]
        self._dispatch_pending_replot()

    def _on_reset_params(self, orig_data: list):
        """Queue all parameter resets and dispatch an asynchronous replot."""
        if self._plotter is None:
            return
        self._pending_param_changes = list(orig_data)
        self._dispatch_pending_replot()

    def _dispatch_pending_replot(self):
        """Start a ReplotWorker if nothing is currently running."""
        if self._plotter is None or self._pending_param_changes is None:
            return
        if (self._worker and self._worker.isRunning()) or \
                (self._replot_worker and self._replot_worker.isRunning()):
            return  # a current run will call us again when it finishes
        changes = self._pending_param_changes
        self._pending_param_changes = None
        self._replot_worker = ReplotWorker(self._plotter, changes)
        self._replot_worker.finished.connect(self._on_replot_finished)
        self._replot_worker.error_occurred.connect(self._on_replot_error)
        self.sliders_tab.setEnabled(False)
        self._replot_worker.start()

    def _on_replot_finished(self, spectra, residuals, component_curves, stat, dof):
        self._last_plot_data = (spectra, residuals, component_curves, stat, dof)
        self._replot()
        self.sliders_tab.setEnabled(True)
        # Drain any slider change that arrived while we were busy
        self._dispatch_pending_replot()

    def _on_replot_error(self, msg: str):
        self.log_view.append(f"<span style='color:red'><b>Slider error:</b></span> {msg}")
        self.statusBar().showMessage("Slider update error — see log for details.")
        self.sliders_tab.setEnabled(True)
        self._dispatch_pending_replot()

    def _capture_params_to_xcm(self):
        """Write current AllModels parameter values to <stem>_XiP_params.xcm."""
        if self._plotter is None:
            self.statusBar().showMessage("No model loaded.")
            return
        if (self._worker and self._worker.isRunning()) or \
                (self._replot_worker and self._replot_worker.isRunning()):
            self.statusBar().showMessage("Cannot save while analysis is running.")
            return
        xcm_path = self._plotter.xcm_path
        out_name = xcm_path.stem + "_XiP_params.xcm"
        out_path = xcm_path.parent / out_name
        cwd = os.getcwd()
        try:
            os.chdir(xcm_path.parent)
            # Remove pre-existing file so XSPEC never prompts "replace?"
            if out_path.exists():
                out_path.unlink()
            Xset.save(out_name)
        except Exception as exc:
            self.statusBar().showMessage(f"Save failed: {exc}")
            return
        finally:
            os.chdir(cwd)
        self.statusBar().showMessage(f"Saved current parameters to {out_name}")

    def _on_error(self, msg: str):
        self.log_view.append(f"<span style='color:red'><b>ERROR:</b></span> {msg}")
        self.plot_btn.setEnabled(True)
        self.plot_btn.setText("Plot")
        self.sliders_tab.setEnabled(True)
        self.statusBar().showMessage("Error — see log for details.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("XiP")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
