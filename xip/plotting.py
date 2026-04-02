"""
plotting.py
Shared rendering logic for XiP — used by both the CLI backend and the Qt GUI canvas.

Public API
----------
KIND_STYLES      dict  — colour/linestyle map for the five spectrum roles.
render_xspec_plot(ax1, ax2, spectra, residuals_by_spec, component_curves,
                  stat, dof, modes=None, profile="eROSITA", energy_range=None)

The *profile* argument accepts either a profile name string (``"eROSITA"``,
``"Default"``) **or** a :class:`~xip.profiles.base.XiPProfile` instance.
"""

import numpy as np
import matplotlib.pyplot as plt  # noqa: F401 (imported for callers' convenience)

from xip.profiles import load_profiles
from xip.profiles.base import XiPProfile

# Build a name→profile lookup once at import time.
_PROFILES: dict[str, "XiPProfile"] = {p.name: p for p in load_profiles()}
_FALLBACK_PROFILE_NAME = "Default"


# ---------------------------------------------------------------------------
# Shared style map
# ---------------------------------------------------------------------------

KIND_STYLES: dict[str, dict] = {
    "source":    {"color": "black",      "linestyle": "-",  "label": "Source data"},
    "bkg":       {"color": "teal",       "linestyle": "-",  "label": "Background data"},
    "astro_bkg": {"color": "teal",       "linestyle": "--", "label": "Astrophysical background"},
    "inst_bkg":  {"color": "tab:green",  "linestyle": ":",  "label": "Instrumental background"},
    "other":     {"color": "tab:purple", "linestyle": "-.", "label": "Model"},
}


# ---------------------------------------------------------------------------
# Core render function
# ---------------------------------------------------------------------------

def render_xspec_plot(
    ax1,
    ax2,
    spectra: dict,
    residuals_by_spec: dict[int, dict],
    component_curves: dict,
    stat,
    dof,
    modes=None,
    profile=None,
    energy_range=None,
):
    """Draw the counts panel (*ax1*) and residuals panel (*ax2*).

    Parameters
    ----------
    ax1, ax2          : matplotlib Axes — counts panel and residuals panel.
    spectra           : dict with keys "source", "bkg", "others".
    residuals_by_spec : dict[spec_id → residuals dict] from _collect_plot_arrays().
    component_curves  : dict with keys "source", "astro_bkg", "inst_bkg", "other".
    stat, dof         : fit statistic and degrees of freedom (may be None).
    modes             : set of ints controlling which panels to draw (default {1}).
                        The meaning of each mode number is defined by the profile.
    profile           : a :class:`~xip.profiles.base.XiPProfile` instance **or**
                        a profile name string (e.g. ``"eROSITA"``).
                        Defaults to ``"eROSITA"`` when *None* is passed.
    energy_range      : (emin, emax) tuple used to set x-axis limits.
    """
    # Resolve profile: accept name string or object
    if profile is None:
        profile = _PROFILES.get("eROSITA", _PROFILES.get(_FALLBACK_PROFILE_NAME))
    elif isinstance(profile, str):
        profile = _PROFILES.get(profile, _PROFILES.get(_FALLBACK_PROFILE_NAME))
    # profile is now an XiPProfile instance

    if not modes:
        modes = {1}

    ax1.cla()
    ax2.cla()
    ax2.sharex(ax1)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.tick_params(axis="x", which="both", labelbottom=False)

    # Build a fast lookup: spec_id → spectrum dict
    spec_lookup: dict[int, dict] = {}
    for key in ("source", "bkg"):
        sp = spectra.get(key)
        if sp:
            spec_lookup[sp["id"]] = sp
    for other in spectra.get("others", []):
        spec_lookup[other["id"]] = other

    # Ask the profile which source spec_ids are visible and what colors/labels
    # to use.  Both default to "show all with default style" for non-NuSTAR profiles.
    _vis_ids = profile.visible_source_spec_ids(modes, spec_lookup) if profile else None
    _src_info = profile.source_spec_info(spec_lookup) if profile else {}

    def _src_visible(spec_id: int) -> bool:
        return _vis_ids is None or spec_id in _vis_ids

    def _src_color(spec_id: int) -> str:
        return _src_info.get(spec_id, {}).get("color", KIND_STYLES["source"]["color"])

    def _src_data_label(spec_id: int, fallback: str) -> str:
        return _src_info.get(spec_id, {}).get("label", fallback)

    def _src_model_label(spec_id: int, fallback: str) -> str:
        info = _src_info.get(spec_id)
        return (info["label"] + " model") if info else fallback

    # ── Data points ──────────────────────────────────────────────────────────
    plotted_y = []
    if 1 in modes or 2 in modes:
        sp = spectra.get("source")
        if sp is not None and sp["y"].size > 0 and _src_visible(sp["id"]):
            ax1.errorbar(
                sp["x"], sp["y"],
                xerr=sp["xerr"], yerr=sp["yerr"],
                fmt=".", color=_src_color(sp["id"]),
                label=_src_data_label(sp["id"], KIND_STYLES["source"]["label"]),
                alpha=0.6, lw=1.2,
            )
            plotted_y.append(sp["y"])

        # Additional source-kind spectra (e.g. NuSTAR FPMB) sit in "others".
        extra_src_counter = 2
        for other_sp in spectra.get("others", []):
            if other_sp.get("kind") != "source" or other_sp["y"].size == 0:
                continue
            if not _src_visible(other_sp["id"]):
                continue
            ax1.errorbar(
                other_sp["x"], other_sp["y"],
                xerr=other_sp["xerr"], yerr=other_sp["yerr"],
                fmt=".", color=_src_color(other_sp["id"]),
                label=_src_data_label(other_sp["id"], f"Source data ({extra_src_counter})"),
                alpha=0.6, lw=1.2,
            )
            plotted_y.append(other_sp["y"])
            extra_src_counter += 1

    if 3 in modes:
        sp = spectra.get("bkg")
        if sp is not None and sp["y"].size > 0:
            style = KIND_STYLES["bkg"]
            ax1.errorbar(
                sp["x"], sp["y"],
                xerr=sp["xerr"], yerr=sp["yerr"],
                fmt=".", color=style["color"],
                label=style["label"], alpha=0.6, lw=1.2,
            )
            plotted_y.append(sp["y"])

    if plotted_y:
        all_y = np.concatenate(plotted_y)
        ax1.set_ylim(
            max(np.min(all_y) / 5, 1e-6),
            np.max(all_y) * 5,
        )

    # ── Total model curves ────────────────────────────────────────────────────
    # Determine which mode numbers involve the source / bkg spectrum.
    # We ask the profile which modes it declares and check membership.
    profile_mode_nums = {m for m, _ in profile.modes} if profile else {1, 2, 3}
    # Source model: shown whenever any "source" mode is active
    source_modes = profile_mode_nums - {3}  # everything except the bkg-only mode
    if modes & source_modes:
        src_sp = spectra.get("source")
        if src_sp is not None and _src_visible(src_sp["id"]):
            ax1.plot(
                src_sp["x"], src_sp["model"],
                color=_src_color(src_sp["id"]),
                linestyle=KIND_STYLES["source"]["linestyle"],
                lw=1.8, label=_src_model_label(src_sp["id"], "Source model"),
            )
        # Model curves for extra source-kind spectra (e.g. NuSTAR FPMB)
        extra_model_counter = 2
        for other_sp in spectra.get("others", []):
            if other_sp.get("kind") != "source":
                continue
            if not _src_visible(other_sp["id"]):
                continue
            ax1.plot(
                other_sp["x"], other_sp["model"],
                color=_src_color(other_sp["id"]),
                linestyle=KIND_STYLES["source"]["linestyle"],
                lw=1.8, label=_src_model_label(other_sp["id"], f"Source model ({extra_model_counter})"),
            )
            extra_model_counter += 1

    if 3 in modes:
        bkg_sp = spectra.get("bkg")
        if bkg_sp is not None:
            style = KIND_STYLES["astro_bkg"]
            ax1.plot(
                bkg_sp["x"], bkg_sp["model"],
                color=style["color"], linestyle=style["linestyle"],
                lw=1.6, label="Background model",
            )

    # ── Component curves — delegated to the active profile ────────────────────
    if profile is not None:
        profile.render_components(ax1, component_curves, spec_lookup, modes, KIND_STYLES)

    # ── Counts-panel formatting ───────────────────────────────────────────────
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylabel(r"$\mathrm{Counts\ s^{-1}\ keV^{-1}}$")

    handles, labels = ax1.get_legend_handles_labels()
    seen: dict = {}
    for h, lbl in zip(handles, labels):
        if lbl not in seen:
            seen[lbl] = h
    ax1.legend(
        seen.values(), seen.keys(),
        bbox_to_anchor=(1.01, 1.0), loc="upper left",
        fontsize=8, borderaxespad=0,
    )

    if stat is not None and dof:
        title = f"CSTAT/d.o.f. = {stat / dof:.2f}  ({stat:.2f} / {dof})"
    else:
        title = "XSPEC model vs data"
    ax1.set_title(title)

    # ── Residuals panel ───────────────────────────────────────────────────────
    # Ask the profile which spec_ids should contribute residuals.
    # None means "use the standard single-spectrum fallback".
    _resid_ids = profile.visible_residual_spec_ids(modes, spec_lookup) if profile else None

    resid_entries: list[tuple[dict, str]] = []  # (resid_dict, color)

    if _resid_ids is not None:
        # Profile-driven: draw residuals for every requested spec_id.
        for spec_id in sorted(_resid_ids):
            r = residuals_by_spec.get(spec_id)
            if r is None or r["x"].size == 0:
                continue
            color = _src_color(spec_id)
            resid_entries.append((r, color))
    else:
        # Default fallback: one residual series chosen by visibility.
        resid: dict | None = None
        if 3 in modes and 1 not in modes and 2 not in modes:
            bkg_sp = spectra.get("bkg")
            if bkg_sp is not None:
                resid = residuals_by_spec.get(bkg_sp["id"])
        if resid is None:
            src_sp = spectra.get("source")
            if src_sp is not None:
                resid = residuals_by_spec.get(src_sp["id"])
        if resid is None and residuals_by_spec:
            resid = next(iter(residuals_by_spec.values()))
        if resid is not None and resid["x"].size:
            resid_entries.append((resid, "black"))

    for resid, resid_color in resid_entries:
        ax2.errorbar(
            resid["x"], resid["y"],
            xerr=resid["xerr"], yerr=resid["yerr"],
            fmt=".", color=resid_color, alpha=0.6, lw=1.2,
        )

    ax2.axhline(0, color="gray", linestyle="--", lw=1.0)
    ax2.set_xscale("log")
    ax2.set_xlabel("Energy [keV]")
    ax2.set_ylabel("Residuals")

    _CANDIDATE_TICKS = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 80]
    if energy_range is not None:
        emin, emax = energy_range
        ax2.set_xlim(emin, emax)
    else:
        # Derive visible range from the actual data present
        xdata: list[float] = []
        for sp in ([spectra.get("source"), spectra.get("bkg")] +
                   spectra.get("others", [])):
            if sp is not None and sp["x"].size:
                xdata.extend([float(sp["x"].min()), float(sp["x"].max())])
        emin, emax = (min(xdata), max(xdata)) if xdata else (0.1, 10.0)
    _ticks = [t for t in _CANDIDATE_TICKS if emin <= t <= emax]
    if len(_ticks) < 3:
        # Fallback to evenly-spaced log ticks when candidates are sparse
        _ticks_raw = np.logspace(np.log10(emin), np.log10(emax), 5)
        _ticks = [round(float(t), 4) for t in _ticks_raw]
    ax2.set_xticks(_ticks)
    ax2.set_xticklabels([str(t) for t in _ticks])
    # ── Residuals y-axis: auto-scale to actual data range, symmetric ────────
    all_resid_y = np.concatenate(
        [r["y"] for r, _ in resid_entries if r["y"].size > 0]
    ) if resid_entries else np.array([])
    rvals = all_resid_y[np.isfinite(all_resid_y)] if all_resid_y.size else np.array([])
    if rvals.size:
        spread = max(abs(float(np.min(rvals))), abs(float(np.max(rvals))))
        pad = spread * 0.3 or 1.0
        ylim = min(spread + pad, 10.0)
    else:
        ylim = 5.0
    # Symmetric axis keeps the zero line centred; tick at ±⌊0.65 * ylim⌋
    ylim = max(ylim, 1.0)  # never collapse to zero
    tick_val = max(1.0, round(ylim * 0.65))
    ax2.set_ylim(-ylim, ylim)
    ax2.set_yticks([-tick_val, 0, tick_val])
    ax2.axhspan(-3, 3, color="gray", alpha=0.15)
