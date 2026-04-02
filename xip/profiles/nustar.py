"""NuSTAR instrument profile for XiP.

Two-detector single-model layout: FPMA and FPMB share one XSPEC model.
Both detectors are source spectra; there is no separate background model
(the background is folded into the spectral model, often via a constant
cross-normalisation factor between FPMA and FPMB).

Modes
-----
1  FPMA only — data points and source model components for the first detector
2  FPMB only — data points and source model components for the second detector

Both modes are enabled by default so both detectors are shown simultaneously.
"""

from xip.profiles.base import XiPProfile

# Data points + total model curve colors — deliberately match the eROSITA/Default
# convention: FPMA uses black (primary source), FPMB uses teal (secondary source).
_FPMA_DATA_COLOR = "black"
_FPMB_DATA_COLOR = "teal"

# Component curve color cycles — each detector gets its own distinct palette so
# their components never share a color, even when both modes are active.
_FPMA_COMP_COLORS = ["tab:blue",   "tab:orange", "tab:green",  "tab:red"]
_FPMB_COMP_COLORS = ["tab:purple", "tab:brown",  "tab:pink",   "tab:gray"]


class NuSTARProfile(XiPProfile):
    """Profile for NuSTAR data with two source detectors sharing one model."""

    @property
    def name(self) -> str:
        return "NuSTAR"

    @property
    def modes(self) -> list[tuple[int, str]]:
        return [
            (1, "FPMA: data + source components"),
            (2, "FPMB: data + source components"),
        ]

    @property
    def default_modes(self) -> set[int]:
        return {1, 2}

    def detect(self, model_kind_map: dict[str, str]) -> bool:
        # NuSTAR files have no reliable model-kind signature that distinguishes
        # them from a generic single-source fit, so we do not auto-select.
        # The user picks this profile from the dropdown after loading.
        return False

    def _mode_to_spec_id(self, spec_lookup: dict[int, dict]) -> dict[int, int]:
        """Map mode number → spec_id for source-kind spectra."""
        source_ids = sorted(
            sid for sid, sp in spec_lookup.items() if sp.get("kind") == "source"
        )
        slot: dict[int, int] = {}
        if len(source_ids) >= 1:
            slot[1] = source_ids[0]
        if len(source_ids) >= 2:
            slot[2] = source_ids[1]
        return slot

    def visible_source_spec_ids(
        self, modes: set[int], spec_lookup: dict[int, dict]
    ) -> set[int]:
        """Mode 1 → FPMA (first source spec_id), Mode 2 → FPMB (second)."""
        slot = self._mode_to_spec_id(spec_lookup)
        return {spec_id for mode_num, spec_id in slot.items() if mode_num in modes}

    def visible_residual_spec_ids(
        self, modes: set[int], spec_lookup: dict[int, dict]
    ) -> set[int]:
        """Show residuals for whichever detectors are currently selected."""
        return self.visible_source_spec_ids(modes, spec_lookup)

    def source_spec_info(
        self, spec_lookup: dict[int, dict]
    ) -> dict[int, dict]:
        """Fixed data/model colors and labels for FPMA (black) and FPMB (teal).

        Also carries ``comp_colors`` — a per-detector palette used by
        :meth:`render_components` for individual component curves.
        """
        source_ids = sorted(
            sid for sid, sp in spec_lookup.items() if sp.get("kind") == "source"
        )
        _data_colors  = [_FPMA_DATA_COLOR,  _FPMB_DATA_COLOR]
        _comp_palettes = [_FPMA_COMP_COLORS, _FPMB_COMP_COLORS]
        _labels = ["FPMA", "FPMB"]
        result: dict[int, dict] = {}
        for i, sid in enumerate(source_ids):
            result[sid] = {
                "color":       _data_colors[i]   if i < len(_data_colors)   else "tab:cyan",
                "comp_colors": _comp_palettes[i] if i < len(_comp_palettes) else _FPMA_COMP_COLORS,
                "label":       _labels[i]         if i < len(_labels)        else f"FPM{i + 1}",
            }
        return result

    def render_components(self, ax, component_curves, spec_lookup, modes, kind_styles):
        """Draw source components for the selected detector(s).

        Colors and visibility are driven by :meth:`source_spec_info` and
        :meth:`visible_source_spec_ids` so they stay consistent with the
        data points and model curves drawn by the plotting layer.
        """
        draw_ids = self.visible_source_spec_ids(modes, spec_lookup)
        if not draw_ids:
            return

        det_info = self.source_spec_info(spec_lookup)

        # Track how many components have been plotted per detector so we can
        # cycle through each detector's distinct color palette independently.
        comp_counters: dict[int, int] = {}  # spec_id -> index into comp_colors
        for comp in component_curves.get("source", []):
            spec_id = comp["spec_id"]
            if spec_id not in draw_ids:
                continue
            sp = spec_lookup.get(spec_id)
            if sp is None:
                continue
            info = det_info.get(spec_id, {})
            palette = info.get("comp_colors", _FPMA_COMP_COLORS)
            det_label = info.get("label", f"det{spec_id}")
            idx = comp_counters.get(spec_id, 0)
            color = palette[idx % len(palette)]
            comp_counters[spec_id] = idx + 1
            lbl_text = comp.get("component_label")
            if lbl_text:
                lbl = f"{det_label}: {comp['model_name']}: {lbl_text}"
            else:
                lbl = f"{det_label}: component {idx + 1}"
            ax.plot(sp["x"], comp["curve"], color=color, linestyle="-", lw=1.5, label=lbl)


PROFILE = NuSTARProfile()
