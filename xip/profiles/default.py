"""Default (generic) instrument profile for XiP.

Works for any one- or two-model layout (source only, or source + background).
All component kinds are drawn on whichever spectrum they belong to.

Modes
-----
1  Source spectrum — data + source model + all components
3  Background spectrum — data + background model + all background components
"""

from xip.profiles.base import XiPProfile


class DefaultProfile(XiPProfile):
    """Generic profile for non-eROSITA data (NuSTAR, XMM, Chandra, Suzaku …)."""

    @property
    def name(self) -> str:
        return "Default"

    @property
    def modes(self) -> list[tuple[int, str]]:
        return [
            (1, "Source: data + all components"),
            (2, "Background: data + model"),
        ]

    @property
    def default_modes(self) -> set[int]:
        return {1}

    def detect(self, model_kind_map: dict[str, str]) -> bool:
        # Fallback — never auto-selected; used when no other profile matches.
        return False

    def render_components(self, ax, component_curves, spec_lookup, modes, kind_styles):
        # ── Mode 1: all components on the source spectrum ─────────────────────
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

            extra_counter = 1
            for kind in ("astro_bkg", "inst_bkg", "other"):
                for comp in component_curves.get(kind, []):
                    sp = spec_lookup.get(comp["spec_id"])
                    if sp is None or sp.get("kind") != "source":
                        continue
                    fallback = f"component {extra_counter}"
                    extra_counter += 1
                    lbl_text = comp.get("component_label")
                    lbl = f"{comp['model_name']}: {lbl_text}" if lbl_text else fallback
                    ls = kind_styles.get(kind, kind_styles["other"])["linestyle"]
                    ax.plot(sp["x"], comp["curve"], linestyle=ls, lw=1.0, label=lbl)

        # ── Mode 2: all bkg components on the background spectrum ─────────────
        if 2 in modes:
            bkg_counter = 1
            for kind in ("astro_bkg", "inst_bkg", "other"):
                for comp in component_curves.get(kind, []):
                    sp = spec_lookup.get(comp["spec_id"])
                    if sp is None or sp.get("kind") != "bkg":
                        continue
                    fallback = f"bkg component {bkg_counter}"
                    bkg_counter += 1
                    lbl_text = comp.get("component_label")
                    lbl = f"{comp['model_name']}: {lbl_text}" if lbl_text else fallback
                    ax.plot(sp["x"], comp["curve"], linestyle="--", lw=1.0, label=lbl)


PROFILE = DefaultProfile()
