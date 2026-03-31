"""eROSITA instrument profile for XiP.

Three-model layout:  source / astrophysical background / instrumental background.

Modes
-----
1  Source spectrum — data + source model + source components
2  Source spectrum — data + source model + background components projected onto source
3  Background spectrum — data + background model + background components
"""

from xip.profiles.base import XiPProfile


class EROSITAProfile(XiPProfile):
    """Profile for eROSITA data with a three-model (src / astro_bkg / inst_bkg) setup."""

    @property
    def name(self) -> str:
        return "eROSITA"

    @property
    def modes(self) -> list[tuple[int, str]]:
        return [
            (1, "Source: data + source components"),
            (2, "Source: data + bkg components"),
            (3, "Background: data + model"),
        ]

    @property
    def default_modes(self) -> set[int]:
        return {1}

    def detect(self, model_kind_map: dict[str, str]) -> bool:
        """Auto-select when an instrumental background model is present."""
        return "inst_bkg" in model_kind_map.values()

    def render_components(self, ax, component_curves, spec_lookup, modes, kind_styles):
        # ── Mode 1: source components on the source spectrum ─────────────────
        if 1 in modes:
            src_counter = 1
            for comp in component_curves.get("source", []):
                sp = spec_lookup.get(comp["spec_id"])
                if sp is None:
                    continue
                fallback = f"src component {src_counter}"
                src_counter += 1
                lbl_text = comp.get("component_label")
                lbl = f"{comp['model_name']}: {lbl_text}" if lbl_text else fallback
                ax.plot(sp["x"], comp["curve"], linestyle="-", lw=1.5, label=lbl)

        # ── Mode 2: bkg components projected onto the source spectrum ─────────
        if 2 in modes:
            astro_counter = 1
            for comp in component_curves.get("astro_bkg", []):
                sp = spec_lookup.get(comp["spec_id"])
                if sp is None or sp.get("kind") != "source":
                    continue
                fallback = f"bkg component {astro_counter}"
                astro_counter += 1
                lbl_text = comp.get("component_label")
                lbl = f"{comp['model_name']}: {lbl_text}" if lbl_text else fallback
                ax.plot(sp["x"], comp["curve"], linestyle="--", lw=1.0, label=lbl)
            self._plot_inst_bkg_summed(
                ax, component_curves, spec_lookup, kind_styles, target_kind="source"
            )

        # ── Mode 3: bkg components on the background spectrum ─────────────────
        if 3 in modes:
            astro_counter = 1
            for comp in component_curves.get("astro_bkg", []):
                sp = spec_lookup.get(comp["spec_id"])
                if sp is None or sp.get("kind") != "bkg":
                    continue
                fallback = f"bkg component {astro_counter}"
                astro_counter += 1
                lbl_text = comp.get("component_label")
                lbl = f"{comp['model_name']}: {lbl_text}" if lbl_text else fallback
                ax.plot(sp["x"], comp["curve"], linestyle="--", lw=1.0, label=lbl)
            self._plot_inst_bkg_summed(
                ax, component_curves, spec_lookup, kind_styles, target_kind="bkg"
            )

    # ------------------------------------------------------------------
    def _plot_inst_bkg_summed(self, ax, component_curves, spec_lookup,
                               kind_styles, target_kind: str) -> None:
        """Sum all inst_bkg curves for each (spec_id, model_name) group
        that belongs to *target_kind* spectrum and plot as a single line."""
        import numpy as np
        from collections import defaultdict

        style = kind_styles["inst_bkg"]
        # Group curves by (spec_id, model_name) — sum within each group.
        groups: dict = defaultdict(list)
        for comp in component_curves.get("inst_bkg", []):
            sp = spec_lookup.get(comp["spec_id"])
            if sp is None or sp.get("kind") != target_kind:
                continue
            groups[(comp["spec_id"], comp["model_name"])].append(comp["curve"])

        plotted_labels: set = set()
        for (spec_id, model_name), curves in groups.items():
            sp = spec_lookup[spec_id]
            combined = np.sum(curves, axis=0)
            lbl = style["label"] if style["label"] not in plotted_labels else "_nolegend_"
            plotted_labels.add(style["label"])
            ax.plot(sp["x"], combined,
                    linestyle=style["linestyle"], lw=1.2, label=lbl)


PROFILE = EROSITAProfile()
