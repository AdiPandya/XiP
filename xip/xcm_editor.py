"""
xcm_param_editor.py
Structured parameter editor for XSPEC XCM files.

Public API
----------
parse_xcm_text(text: str)          -> ParsedXCM
serialize_xcm(parsed: ParsedXCM)  -> str
XCMParamEditorWidget(QWidget)       embeddable widget
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from dataclasses import dataclass, field

from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

# ──────────────────────────────────────────────────────────────────────────────
# XSPEC model database  —  loaded from xspec_model_db.json
# ──────────────────────────────────────────────────────────────────────────────
_DB_PATH = Path(__file__).parent / "xspec_model_db.json"
with _DB_PATH.open() as _f:
    _DB = json.load(_f)

_XSPEC_PARAMS: dict[str, list[str]]  = {k.lower(): v for k, v in _DB["models"].items()}
_XSPEC_PARAM_DEFAULTS: dict[str, float] = {k: v["default"] for k, v in _DB["param_info"].items() if isinstance(v, dict)}
_XSPEC_FROZEN_PARAMS: frozenset[str]    = frozenset(k for k, v in _DB["param_info"].items() if isinstance(v, dict) and v["frozen"])
_XSPEC_PARAM_INFO: dict[str, dict]      = {k: v for k, v in _DB["param_info"].items() if isinstance(v, dict)}


def _get_param_names(comp_name: str) -> list[str] | None:
    return _XSPEC_PARAMS.get(comp_name.lower())


def _default_param(name: str) -> ParsedParam:
    """Create a ParsedParam with XSPEC-typical defaults for *name*."""
    info      = _XSPEC_PARAM_INFO.get(name, {})
    value     = info.get("default",  0.0)
    frozen    = info.get("frozen",   False)
    delta_abs = info.get("delta",    0.01)
    delta     = -delta_abs if frozen else delta_abs
    hard_min  = info.get("hard_min", 0.0)
    soft_min  = info.get("soft_min", hard_min)
    soft_max  = info.get("soft_max", 1e6)
    hard_max  = info.get("hard_max", soft_max)
    return ParsedParam(name=name, value=value, delta=delta,
                       hard_min=hard_min, soft_min=soft_min,
                       soft_max=soft_max, hard_max=hard_max)


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ParsedParam:
    name: str
    value: float = 0.0
    delta: float = 0.01
    hard_min: float = 0.0
    soft_min: float = 0.0
    soft_max: float = 1e6
    hard_max: float = 1e6
    is_linked: bool = False
    link_str: str = ""

    @property
    def frozen(self) -> bool:
        return self.delta < 0

    def fmt(self) -> str:
        """Serialise to a 6-column XCM parameter row."""
        if self.is_linked:
            return f"        {self.link_str}"
        vals = [self.value, self.delta, self.hard_min, self.soft_min,
                self.soft_max, self.hard_max]
        return "".join(f"{v:14.6g}" for v in vals)


@dataclass
class ParsedComponent:
    name: str
    display_name: str
    params: list[ParsedParam] = field(default_factory=list)


@dataclass
class ParsedModel:
    source_num: int
    model_name: str
    expression: str
    model_line: str
    components: list[ParsedComponent] = field(default_factory=list)
    pre_lines: list[str] = field(default_factory=list)
    # When a model applies to multiple data groups the param block repeats once
    # per group (XSPEC convention).  data_groups holds per-group component lists;
    # components is always data_groups[0] for backward compatibility.  Empty
    # when the model has a single data group.
    data_groups: list[list[ParsedComponent]] = field(default_factory=list)


@dataclass
class ParsedXCM:
    header_lines: list[str] = field(default_factory=list)
    models: list[ParsedModel] = field(default_factory=list)
    # Each entry: {"type": "data"|"bkg", "group": int, "source_num": int, "filename": str}
    spectra_info: list[dict] = field(default_factory=list)
    # Lines that appear after the last model block (e.g. "bayes off")
    trailer_lines: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Expression parser  —  TBabs(vapec + powerlaw) → ["TBabs", "vapec", "powerlaw"]
# ──────────────────────────────────────────────────────────────────────────────

def _tokenize_expr(expr: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    while i < len(expr):
        c = expr[i]
        if c in "()*+":
            tokens.append(c)
            i += 1
        elif c == " ":
            i += 1
        else:
            j = i
            while j < len(expr) and expr[j] not in "()*+":
                j += 1
            tok = expr[i:j].strip()
            if tok:
                tokens.append(tok)
            i = j
    return tokens


def _parse_tokens(tokens: list[str], pos: int, result: list[str]) -> int:
    while pos < len(tokens):
        tok = tokens[pos]
        if tok == ")":
            return pos + 1
        elif tok in ("+", "*"):
            pos += 1
        elif tok == "(":
            pos = _parse_tokens(tokens, pos + 1, result)
        else:
            result.append(tok)
            pos += 1
            if pos < len(tokens) and tokens[pos] == "(":
                pos = _parse_tokens(tokens, pos + 1, result)
    return pos


def parse_expression_components(expression: str) -> list[str]:
    """Return flat ordered component list from a model expression string."""
    expr = re.sub(r"<\d+>", "", expression)  # strip XSPEC internal numbering
    tokens = _tokenize_expr(expr)
    result: list[str] = []
    _parse_tokens(tokens, 0, result)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# XCM text parser
# ──────────────────────────────────────────────────────────────────────────────

_MODEL_LINE_RE = re.compile(
    r"^\s*model\s+(\d+):(\w+)\s+(.+?)\s*$", re.IGNORECASE
)
# Anonymous model line: "model  expression" (no N:name prefix) 
_ANON_MODEL_LINE_RE = re.compile(
    r"^\s*model\s+(?!\d+:)(.+?)\s*$", re.IGNORECASE
)
_FLOAT_RE = re.compile(r"^[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?$")
# Patterns for extracting data file references from the XCM header
_DATA_LINE_RE  = re.compile(r'^data\s+(.+)$', re.IGNORECASE)
_BGRND_LINE_RE = re.compile(r'^backgrnd\s+(.+)$', re.IGNORECASE)
_DATA_ENTRY_RE = re.compile(r'(\d+):(\d+)\s+(\S+)')


def _is_param_line(line: str) -> bool:
    parts = line.strip().split()
    if len(parts) != 6:
        return False
    return all(_FLOAT_RE.match(p) for p in parts)


def _parse_param_values(line: str) -> ParsedParam:
    vals = [float(p) for p in line.strip().split()]
    return ParsedParam(
        name="",
        value=vals[0], delta=vals[1],
        hard_min=vals[2], soft_min=vals[3],
        soft_max=vals[4], hard_max=vals[5],
    )


def _assign_param_names(
    comp_names: list[str], params: list[ParsedParam]
) -> list[ParsedComponent]:
    """Map flat params list to components using the parameter database.

    Strategy: determine param counts left-to-right (known components) with
    unknown components getting the residual split evenly.
    """
    counts: list[int | None] = [
        len(_get_param_names(n)) if _get_param_names(n) is not None else None
        for n in comp_names
    ]
    known_total = sum(c for c in counts if c is not None)
    n_unknown = counts.count(None)
    leftover = len(params) - known_total

    if n_unknown:
        base, rem = divmod(max(leftover, 0), n_unknown)
        u_idx = 0
        for i, c in enumerate(counts):
            if c is None:
                counts[i] = base + (1 if u_idx < rem else 0)
                u_idx += 1

    result: list[ParsedComponent] = []
    comp_seen: dict[str, int] = {}
    param_idx = 0
    for comp_name, n in zip(comp_names, counts):
        n = n or 0
        lo = comp_name.lower()
        comp_seen[lo] = comp_seen.get(lo, 0) + 1
        display = f"{comp_name} ({comp_seen[lo]})" if comp_seen[lo] > 1 else comp_name
        known = _get_param_names(comp_name) or []
        chunk = params[param_idx: param_idx + n]
        for i, p in enumerate(chunk):
            p.name = known[i] if i < len(known) else f"param_{param_idx + i + 1}"
        param_idx += n
        result.append(ParsedComponent(name=comp_name, display_name=display,
                                      params=list(chunk)))

    # Assign any trailing leftover params without a component
    if param_idx < len(params):
        leftover_params = params[param_idx:]
        for i, p in enumerate(leftover_params):
            p.name = f"param_{param_idx + i + 1}"
        result.append(ParsedComponent(name="?", display_name="(unmatched)",
                                      params=leftover_params))
    return result


def _compute_per_group_param_count(
    comp_names: list[str], n_total: int, n_groups: int
) -> int | None:
    """Return the expected number of params per data group, or ``None``.

    When XSPEC applies one model to *n_groups* data groups the param block
    in the XCM file contains ``n_groups × n_per_group`` lines (numeric or
    linked).  This function detects that pattern.

    Returns ``None`` when the split is ambiguous (e.g. only one group, or
    the total count does not divide evenly, or the known param count for
    the expression already uses all params).
    """
    if n_groups <= 1 or n_total == 0:
        return None
    known_per_comp = [
        len(_get_param_names(n))
        for n in comp_names
        if _get_param_names(n) is not None
    ]
    n_known = sum(known_per_comp)
    has_unknown = any(_get_param_names(n) is None for n in comp_names)

    if not has_unknown:
        # All components in the DB — exact divisibility check.
        return n_known if n_total == n_known * n_groups else None
    else:
        # Some components unknown — require clean divisibility and that the
        # known portion fits within one group.
        if n_total % n_groups != 0:
            return None
        candidate = n_total // n_groups
        return candidate if candidate >= n_known else None


def _parse_spectra_info(parsed: ParsedXCM, header_lines: list[str]) -> None:
    """Extract data and backgrnd file references from XCM header lines.

    Handles the common XSPEC formats::

        data 1:1 source.pha 2:1 source2.pha
        backgrnd 1 bkg.pha 2 bkg2.pha
    """
    for line in header_lines:
        stripped = line.strip()
        dm = _DATA_LINE_RE.match(stripped)
        if dm:
            rest = dm.group(1)
            matched = False
            for m in _DATA_ENTRY_RE.finditer(rest):
                parsed.spectra_info.append({
                    "type": "data",
                    "group": int(m.group(1)),
                    "source_num": int(m.group(2)),
                    "filename": m.group(3),
                })
                matched = True
            if not matched:
                # Fallback: plain list of filenames (one per group)
                for i, part in enumerate(rest.split(), start=1):
                    if not part.startswith("#"):
                        parsed.spectra_info.append({
                            "type": "data", "group": i,
                            "source_num": 1, "filename": part,
                        })
            continue
        bm = _BGRND_LINE_RE.match(stripped)
        if bm:
            rest = bm.group(1)
            parts = rest.split()
            # Format: "N filename [N2 filename2 ...]"
            i = 0
            while i + 1 < len(parts):
                try:
                    group = int(parts[i])
                    parsed.spectra_info.append({
                        "type": "bkg", "group": group,
                        "source_num": 0, "filename": parts[i + 1],
                    })
                    i += 2
                except ValueError:
                    i += 1


def _match_model_line(line: str):
    """Try both named and anonymous model-line regexes.

    Returns a tuple ``(source_num, model_name, expression)`` on match,
    or ``None`` if the line is not a model line.
    """
    m = _MODEL_LINE_RE.match(line)
    if m:
        return int(m.group(1)), m.group(2), m.group(3).strip()
    m = _ANON_MODEL_LINE_RE.match(line)
    if m:
        # Assign synthetic defaults that mirror XSPEC's unnamed model 1.
        return 1, "src", m.group(1).strip()
    return None


def parse_xcm_text(text: str) -> ParsedXCM:
    """Parse raw XCM text into a ParsedXCM structure."""
    lines = text.splitlines()
    parsed = ParsedXCM()

    model_indices = [i for i, l in enumerate(lines) if _match_model_line(l) is not None]
    if not model_indices:
        parsed.header_lines = lines
        _parse_spectra_info(parsed, lines)
        return parsed

    parsed.header_lines = lines[: model_indices[0]]
    _parse_spectra_info(parsed, parsed.header_lines)

    for blk, start in enumerate(model_indices):
        is_last = (blk + 1 == len(model_indices))
        end = model_indices[blk + 1] if not is_last else len(lines)
        src_num, mod_name, expression = _match_model_line(lines[start])

        raw_params: list[ParsedParam] = []
        pre_lines: list[str] = []
        trailing: list[str] = []  # non-param lines after last param in this block

        seen_last_param = False
        for line in lines[start + 1: end]:
            stripped = line.strip()
            if _is_param_line(line):
                # Any accumulated trailing lines before a new param line belong
                # to a nested block gap — treat as pre_lines for next iteration.
                trailing.clear()
                seen_last_param = True
                raw_params.append(_parse_param_values(line))
            elif stripped.startswith("="):
                trailing.clear()
                seen_last_param = True
                raw_params.append(ParsedParam(name="", is_linked=True,
                                              link_str=stripped))
            elif not stripped:
                pass  # skip blank lines inside block
            else:
                # Non-param, non-blank: if we've seen params already this is a
                # post-param directive (e.g. "bayes off"); otherwise it's a
                # between-block comment / directive — collect as trailing.
                trailing.append(line)

        # Trailing non-param lines at the end of the last model block are
        # post-model directives ("bayes off", etc.) — save them.
        if is_last and trailing:
            parsed.trailer_lines = trailing

        comp_names = parse_expression_components(expression)

        # ── Multi-group detection ──────────────────────────────────────────
        # When the same model applies to N data groups the XCM param block
        # contains N × per_group lines (numeric or linked).  Detect this and
        # split the flat list into per-group components.
        n_data_groups = len({
            e["group"] for e in parsed.spectra_info if e["type"] == "data"
        }) or 1
        n_per_group = _compute_per_group_param_count(
            comp_names, len(raw_params), n_data_groups
        )

        if n_per_group is not None and n_data_groups > 1:
            # Split raw params into one chunk per data group and assign names
            # independently for each chunk.
            data_groups: list[list[ParsedComponent]] = []
            for g in range(n_data_groups):
                chunk = raw_params[g * n_per_group: (g + 1) * n_per_group]
                data_groups.append(_assign_param_names(comp_names, chunk))
            components = data_groups[0]
        else:
            components = _assign_param_names(comp_names, raw_params)
            data_groups = []

        parsed.models.append(ParsedModel(
            source_num=src_num,
            model_name=mod_name,
            expression=expression,
            model_line=lines[start],
            components=components,
            pre_lines=pre_lines,
            data_groups=data_groups,
        ))

    return parsed


def serialize_xcm(parsed: ParsedXCM) -> str:
    """Reconstruct XCM text from a ParsedXCM (with any edits applied)."""
    out: list[str] = list(parsed.header_lines)
    for model in parsed.models:
        out.extend(model.pre_lines)
        out.append(model.model_line)
        # Emit each data group's params in order; fall back to single-group.
        groups = model.data_groups if model.data_groups else [model.components]
        for group in groups:
            for comp in group:
                for p in comp.params:
                    out.append(p.fmt())
    if parsed.trailer_lines:
        out.extend(parsed.trailer_lines)
    return "\n".join(out)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_float(v: float) -> str:
    if v == 0.0:
        return "0"
    a = abs(v)
    if 1e-4 <= a < 1e7:
        return f"{v:.7g}"
    return f"{v:.5e}"


# ──────────────────────────────────────────────────────────────────────────────
# GUI widgets
# ──────────────────────────────────────────────────────────────────────────────

class _ParamRowWidget(QWidget):
    """One parameter row: name | value edit | Frozen checkbox | delta edit."""

    def __init__(self, param: ParsedParam, parent: QWidget | None = None):
        super().__init__(parent)
        self._orig = param
        self._build(param)

    def _build(self, p: ParsedParam) -> None:
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 1, 2, 1)
        lay.setSpacing(6)

        # Parameter name
        name_lbl = QLabel(p.name)
        name_lbl.setFixedWidth(110)
        name_lbl.setAlignment(Qt.AlignmentFlag.AlignRight |
                               Qt.AlignmentFlag.AlignVCenter)
        f = name_lbl.font()
        f.setFamily("Courier New")
        name_lbl.setFont(f)
        lay.addWidget(name_lbl)

        # Value
        self._val = QLineEdit(_fmt_float(p.value) if not p.is_linked else p.link_str)
        self._val.setFixedWidth(110)
        self._val.setReadOnly(p.is_linked)
        if p.is_linked:
            self._val.setStyleSheet("color: gray; background: #f0f0f0;")
        lay.addWidget(self._val)

        lay.addSpacing(12)

        # Frozen toggle
        self._frz = QCheckBox("Frozen")
        self._frz.setChecked(p.frozen)
        self._frz.setEnabled(not p.is_linked)
        lay.addWidget(self._frz)

        lay.addSpacing(12)

        # Delta
        d_lbl = QLabel("δ:")
        d_lbl.setFixedWidth(18)
        lay.addWidget(d_lbl)
        self._delta = QLineEdit(_fmt_float(abs(p.delta)))
        self._delta.setFixedWidth(80)
        self._delta.setEnabled(not p.is_linked)
        lay.addWidget(self._delta)

        # Bounds (read-only, small)
        bounds_txt = (f"[{_fmt_float(p.hard_min)}, {_fmt_float(p.hard_max)}]"
                      if not p.is_linked else "")
        bounds_lbl = QLabel(bounds_txt)
        bounds_lbl.setStyleSheet("color: #999; font-size: 9px;")
        bounds_lbl.setSizePolicy(QSizePolicy.Policy.Expanding,
                                 QSizePolicy.Policy.Preferred)
        lay.addWidget(bounds_lbl)

    def get_param(self) -> ParsedParam:
        p = ParsedParam(
            name=self._orig.name,
            hard_min=self._orig.hard_min, soft_min=self._orig.soft_min,
            soft_max=self._orig.soft_max, hard_max=self._orig.hard_max,
            is_linked=self._orig.is_linked, link_str=self._orig.link_str,
        )
        if p.is_linked:
            return p
        try:
            p.value = float(self._val.text())
        except ValueError:
            p.value = self._orig.value
        try:
            delta = abs(float(self._delta.text()))
        except ValueError:
            delta = abs(self._orig.delta)
        p.delta = -delta if self._frz.isChecked() else delta
        return p


class _ComponentWidget(QGroupBox):
    """A collapsible group of parameter rows for one XSPEC component."""

    def __init__(self, comp: ParsedComponent, parent: QWidget | None = None):
        super().__init__(comp.display_name, parent)

        f = QFont()
        f.setItalic(True)
        self.setFont(f)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 6, 6)
        lay.setSpacing(2)

        # Column header
        hdr = QWidget()
        hlay = QHBoxLayout(hdr)
        hlay.setContentsMargins(2, 0, 2, 0)
        hlay.setSpacing(6)
        hf = QFont()
        hf.setPointSize(8)
        for txt, w in [("Parameter", 110), ("Value", 110),
                       ("Frozen", 60), ("", 18), ("Step (δ)", 80)]:
            lbl = QLabel(txt)
            if w:
                lbl.setFixedWidth(w)
            lbl.setFont(hf)
            lbl.setStyleSheet("color: #888;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight |
                              Qt.AlignmentFlag.AlignVCenter)
            hlay.addWidget(lbl)
        hlay.addStretch()
        lay.addWidget(hdr)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        lay.addWidget(sep)

        self.row_widgets: list[_ParamRowWidget] = []
        for p in comp.params:
            rw = _ParamRowWidget(p, self)
            self.row_widgets.append(rw)
            lay.addWidget(rw)


class _ModelWidget(QGroupBox):
    """All components belonging to one XSPEC model."""

    def __init__(self, model: ParsedModel, parent: QWidget | None = None):
        title = f"Model {model.source_num}:{model.model_name}"
        super().__init__(title, parent)
        self._model = model

        f = QFont()
        f.setBold(True)
        self.setFont(f)

        self._lay = QVBoxLayout(self)
        self._lay.setSpacing(8)

        # ── Expression editor (label + field on first line, Apply on second) ──
        expr_outer = QVBoxLayout()
        expr_outer.setSpacing(3)

        expr_top = QHBoxLayout()
        expr_lbl = QLabel("Expression:")
        expr_lbl.setFixedWidth(80)
        self._expr_edit = QPlainTextEdit(model.expression)
        self._expr_edit.setFont(QFont("Courier New", 10))
        self._expr_edit.setToolTip(
            "Edit the model expression, then click Apply to update parameters."
        )
        self._expr_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._expr_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self._expr_edit.setFixedHeight(
            self._expr_edit.fontMetrics().lineSpacing() * 3 + 12
        )
        expr_top.addWidget(expr_lbl)
        expr_top.addWidget(self._expr_edit)
        expr_outer.addLayout(expr_top)

        expr_btn_row = QHBoxLayout()
        expr_btn_row.addStretch()
        apply_btn = QPushButton("Apply")
        apply_btn.setFixedWidth(80)
        apply_btn.clicked.connect(self._apply_expression)
        expr_btn_row.addWidget(apply_btn)
        expr_outer.addLayout(expr_btn_row)

        self._lay.addLayout(expr_outer)

        # Error / status label
        self._err_lbl = QLabel("")
        self._err_lbl.setStyleSheet("color: red; font-size: 10px;")
        self._err_lbl.setWordWrap(True)
        self._err_lbl.hide()
        self._lay.addWidget(self._err_lbl)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        self._lay.addWidget(sep)

        self.comp_widgets: list[_ComponentWidget] = []
        # Parallel structure to comp_widgets: one inner list per data group.
        self._group_comp_widgets: list[list[_ComponentWidget]] = []
        self._build_comp_widgets()

    # ------------------------------------------------------------------
    def _build_comp_widgets(self) -> None:
        """Clear and recreate _ComponentWidget children from self._model."""
        for cws in self._group_comp_widgets:
            for cw in cws:
                cw.setParent(None)
        self._group_comp_widgets.clear()
        self.comp_widgets.clear()

        groups = self._model.data_groups if len(self._model.data_groups) > 1 else [self._model.components]

        for g_idx, group_comps in enumerate(groups):
            if len(groups) > 1:
                # Wrap each data group in a labelled QGroupBox.
                grp_box = QGroupBox(f"Data Group {g_idx + 1}")
                grp_f = QFont()
                grp_f.setItalic(True)
                grp_box.setFont(grp_f)
                grp_lay = QVBoxLayout(grp_box)
                grp_lay.setContentsMargins(4, 8, 4, 6)
                grp_lay.setSpacing(4)
                parent_widget = grp_box
                add_to = grp_lay
            else:
                grp_box = None
                parent_widget = self
                add_to = self._lay

            group_cws: list[_ComponentWidget] = []
            for comp in group_comps:
                cw = _ComponentWidget(comp, parent_widget)
                group_cws.append(cw)
                add_to.addWidget(cw)

            self._group_comp_widgets.append(group_cws)
            self.comp_widgets.extend(group_cws)
            if grp_box is not None:
                self._lay.addWidget(grp_box)

    # ------------------------------------------------------------------
    def _flush_values_to_model(self) -> None:
        """Write current widget values back to self._model before any rebuild."""
        groups = self._model.data_groups if len(self._model.data_groups) > 1 else [self._model.components]
        for group_cws, group_comps in zip(self._group_comp_widgets, groups):
            for cw, comp in zip(group_cws, group_comps):
                for rw, param in zip(cw.row_widgets, comp.params):
                    updated = rw.get_param()
                    param.value = updated.value
                    param.delta = updated.delta

    # ------------------------------------------------------------------
    def _apply_expression(self) -> None:
        """Validate the edited expression and rebuild the parameter form."""
        new_expr = self._expr_edit.toPlainText().strip()
        if not new_expr or new_expr == self._model.expression:
            self._err_lbl.hide()
            return

        new_comp_names = parse_expression_components(new_expr)
        if not new_comp_names:
            self._err_lbl.setText("Expression parsed to zero components — check syntax.")
            self._err_lbl.show()
            return

        # Validate every component against the database
        unknown = [n for n in new_comp_names if _get_param_names(n) is None]
        if unknown:
            self._err_lbl.setText(
                f"Unsupported component(s): {', '.join(unknown)}\n"
                "Check the name or add it to _XSPEC_PARAMS in xcm_param_editor.py."
            )
            self._err_lbl.show()
            return

        self._err_lbl.hide()

        # Preserve current edited values before we throw away the widgets
        self._flush_values_to_model()

        # Map old components by lower-case name so we can transfer values
        old_by_name: dict[str, list[ParsedComponent]] = {}
        for comp in self._model.components:
            old_by_name.setdefault(comp.name.lower(), []).append(comp)
        used_old: dict[str, int] = {}

        new_components: list[ParsedComponent] = []
        comp_seen: dict[str, int] = {}
        for comp_name in new_comp_names:
            lo = comp_name.lower()
            comp_seen[lo] = comp_seen.get(lo, 0) + 1
            display = (
                f"{comp_name} ({comp_seen[lo]})"
                if comp_seen[lo] > 1
                else comp_name
            )
            param_names = _get_param_names(comp_name) or []

            # Reuse values from a previous component of the same name
            used_idx = used_old.get(lo, 0)
            old_list = old_by_name.get(lo, [])
            if used_idx < len(old_list):
                old_comp = old_list[used_idx]
                used_old[lo] = used_idx + 1
                new_params: list[ParsedParam] = []
                for i, pname in enumerate(param_names):
                    if i < len(old_comp.params):
                        op = old_comp.params[i]
                        new_params.append(ParsedParam(
                            name=pname,
                            value=op.value, delta=op.delta,
                            hard_min=op.hard_min, soft_min=op.soft_min,
                            soft_max=op.soft_max, hard_max=op.hard_max,
                            is_linked=op.is_linked, link_str=op.link_str,
                        ))
                    else:
                        new_params.append(_default_param(pname))
            else:
                # Completely new component — initialise with database defaults
                new_params = [_default_param(pn) for pn in param_names]

            new_components.append(
                ParsedComponent(name=comp_name, display_name=display,
                                params=new_params)
            )

        # Commit changes to the data model
        self._model.expression = new_expr

        # Update the raw model line — handle both named ("model N:name expr")
        # and anonymous ("model  expr") formats.
        new_line = re.sub(
            r"(model\s+\d+:\w+\s+).+$",
            lambda m, e=new_expr: m.group(1) + e,
            self._model.model_line,
            flags=re.IGNORECASE,
        )
        if new_line == self._model.model_line:
            # Anonymous format: "model  expression"
            new_line = re.sub(
                r"(?i)(^\s*model\s+)(?!\d+:).+$",
                lambda m, e=new_expr: m.group(1) + e,
                self._model.model_line,
            )
        self._model.model_line = new_line

        if len(self._model.data_groups) > 1:
            # Multi-group model: rebuild each group from the new expression.
            # Group 1 uses new_components (values already carried over above).
            # Groups 2+: replicate the component structure but preserve each
            # param's linked/independent status by matching flat param position.
            # Linked params get a recomputed "= p{N}" pointing to group-1 param N.
            new_data_groups: list[list[ParsedComponent]] = [new_components]
            for g_idx in range(1, len(self._model.data_groups)):
                old_group = self._model.data_groups[g_idx]
                old_flat = [p for comp in old_group for p in comp.params]
                extra_group: list[ParsedComponent] = []
                flat_pos = 0  # absolute 0-based index within the group
                for comp_tmpl in new_components:
                    extra_params: list[ParsedParam] = []
                    for p_tmpl in comp_tmpl.params:
                        if flat_pos < len(old_flat):
                            old_p = old_flat[flat_pos]
                            if old_p.is_linked:
                                # Recompute link string to match new group-1
                                # absolute param number (1-based).
                                extra_params.append(ParsedParam(
                                    name=p_tmpl.name,
                                    value=p_tmpl.value, delta=p_tmpl.delta,
                                    hard_min=p_tmpl.hard_min, soft_min=p_tmpl.soft_min,
                                    soft_max=p_tmpl.soft_max, hard_max=p_tmpl.hard_max,
                                    is_linked=True,
                                    link_str=f"= p{flat_pos + 1}",
                                ))
                            else:
                                # Independent param: keep old value.
                                extra_params.append(ParsedParam(
                                    name=p_tmpl.name,
                                    value=old_p.value, delta=old_p.delta,
                                    hard_min=p_tmpl.hard_min, soft_min=p_tmpl.soft_min,
                                    soft_max=p_tmpl.soft_max, hard_max=p_tmpl.hard_max,
                                    is_linked=False, link_str="",
                                ))
                        else:
                            extra_params.append(_default_param(p_tmpl.name))
                        flat_pos += 1
                    extra_group.append(ParsedComponent(
                        name=comp_tmpl.name,
                        display_name=comp_tmpl.display_name,
                        params=extra_params,
                    ))
                new_data_groups.append(extra_group)
            self._model.data_groups = new_data_groups
            self._model.components = new_data_groups[0]
        else:
            self._model.components = new_components
            self._model.data_groups = []

        # Update group-box title
        self.setTitle(
            f"Model {self._model.source_num}:{self._model.model_name}"
        )

        # Rebuild parameter widgets
        self._build_comp_widgets()


# ──────────────────────────────────────────────────────────────────────────────
# Public widget
# ──────────────────────────────────────────────────────────────────────────────

class XCMParamEditorWidget(QWidget):
    """Embeddable structured parameter editor.

    Usage::
        widget = XCMParamEditorWidget()
        widget.load(xcm_text_string)
        ...
        new_xcm_text = widget.get_xcm_text()
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._parsed: ParsedXCM | None = None
        self._model_widgets: list[_ModelWidget] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        outer.addWidget(self._scroll)

        self._content = QWidget()
        self._content.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self._content_lay = QVBoxLayout(self._content)
        self._content_lay.setContentsMargins(6, 6, 6, 6)
        self._content_lay.setSpacing(12)

        placeholder = QLabel("Load an XCM file and switch to Structured mode to edit parameters here.")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: gray; font-size: 13px;")
        self._placeholder = placeholder
        self._content_lay.addWidget(placeholder)
        self._content_lay.addStretch()

        self._scroll.setWidget(self._content)

    def load(self, xcm_text: str) -> None:
        """Parse xcm_text and rebuild the parameter form."""
        # Clear existing model widgets
        for mw in self._model_widgets:
            mw.setParent(None)
        self._model_widgets.clear()

        # Remove trailing stretch
        last = self._content_lay.count() - 1
        if last >= 0:
            item = self._content_lay.takeAt(last)
            if item and item.widget():
                item.widget().setParent(None)
            elif item:
                del item

        self._parsed = parse_xcm_text(xcm_text)

        if not self._parsed.models:
            self._placeholder.show()
            self._content_lay.addWidget(self._placeholder)
            self._content_lay.addStretch()
            return

        self._placeholder.hide()

        # ── Model parameter widgets ──────────────────────────────────────
        for model in self._parsed.models:
            mw = _ModelWidget(model, self._content)
            self._model_widgets.append(mw)
            self._content_lay.addWidget(mw)
        self._content_lay.addStretch()

    def get_xcm_text(self) -> str:
        """Return XCM text with current widget values written back."""
        if self._parsed is None:
            return ""
        for mw in self._model_widgets:
            mw._flush_values_to_model()
        return serialize_xcm(self._parsed)

    def validate_expressions(self) -> list[str]:
        """Check all model expressions against the component database.

        Returns a list of diagnostic strings.  An empty list means all good.
        Lines beginning with '\u2717' are hard errors (expression cannot be parsed).
        Lines beginning with '\u26a0' are warnings (unknown but may still work).
        """
        if self._parsed is None:
            return []
        for mw in self._model_widgets:
            mw._flush_values_to_model()
        return _validate_parsed(self._parsed)


# ──────────────────────────────────────────────────────────────────────────────
# Module-level validation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _validate_parsed(parsed: ParsedXCM) -> list[str]:
    """Internal helper: validate model expressions in a ParsedXCM."""
    errors: list[str] = []
    for model in parsed.models:
        tag = f"Model {model.source_num}:{model.model_name}"
        try:
            comp_names = parse_expression_components(model.expression)
        except Exception as exc:
            errors.append(f"\u2717 {tag} — expression parse error: {exc}")
            continue
        if not comp_names:
            errors.append(
                f"\u2717 {tag} — expression '{model.expression}' "
                "parsed to zero components. Check the syntax."
            )
            continue
        unknown = [n for n in comp_names if _get_param_names(n) is None]
        if unknown:
            errors.append(
                f"\u26a0 {tag} — component(s) not in the local database: "
                f"{', '.join(unknown)}. "
                "They may still be valid XSPEC models."
            )
    return errors


def validate_xcm_text(text: str) -> list[str]:
    """Parse XCM text and validate all model expressions (no widget needed).

    Returns a list of diagnostic strings (same format as
    ``XCMParamEditorWidget.validate_expressions()``).
    """
    try:
        parsed = parse_xcm_text(text)
    except Exception as exc:
        return [f"\u2717 Failed to parse XCM text: {exc}"]
    return _validate_parsed(parsed)
