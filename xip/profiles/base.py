"""Base class for XiP instrument profiles."""

from abc import ABC, abstractmethod


class XiPProfile(ABC):
    """Abstract base class for XiP instrument profiles.

    A profile controls:

    * Which **mode checkboxes** appear in the sidebar (``modes`` property).
    * Which modes are **checked by default** (``default_modes`` property).
    * Whether this profile should be **auto-selected** after a file loads
      (``detect`` method).
    * How **model component curves** are drawn onto the counts panel
      (``render_components`` method).

    Subclasses only need to implement :attr:`name`, :attr:`modes`, and
    :meth:`render_components`.  Override :meth:`detect` for auto-selection.
    """

    # ------------------------------------------------------------------
    # Identity / mode declarations  (override in subclass)
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Short display name shown in the GUI profile drop-down."""
        ...

    @property
    @abstractmethod
    def modes(self) -> list[tuple[int, str]]:
        """Ordered list of ``(mode_number, checkbox_label)`` pairs.

        *mode_number* is an arbitrary positive integer used as a key when
        ``render_components`` is called.  Mode 1 should always be included
        as the primary "source" view.

        Example::

            return [
                (1, "Source: data + source components"),
                (2, "Source: data + background components"),
                (3, "Background: data + model"),
            ]
        """
        ...

    @property
    def default_modes(self) -> set[int]:
        """Set of mode numbers that are checked when this profile is first selected.

        Defaults to the *first* mode declared in :attr:`modes`.
        """
        return {self.modes[0][0]} if self.modes else set()

    # ------------------------------------------------------------------
    # Auto-detection
    # ------------------------------------------------------------------

    def detect(self, model_kind_map: dict[str, str]) -> bool:
        """Return ``True`` to auto-select this profile after a file loads.

        ``model_kind_map`` maps model name → kind string
        (``"source"``, ``"astro_bkg"``, ``"inst_bkg"``, ``"other"``).

        The first profile in the discovery order whose :meth:`detect`
        returns ``True`` is chosen; if none match, the GUI falls back to
        "Default".
        """
        return False

    def visible_source_spec_ids(
        self, modes: set[int], spec_lookup: dict[int, dict]
    ) -> "set[int] | None":
        """Return which source-kind spec_ids should have data + model drawn.

        Return ``None`` (default) to draw **all** source-kind spectra whenever
        any source mode is active.  Override to map specific mode numbers to
        specific detectors.
        """
        return None

    def visible_residual_spec_ids(
        self, modes: set[int], spec_lookup: dict[int, dict]
    ) -> "set[int] | None":
        """Return which spec_ids should have residuals drawn.

        Return ``None`` (default) to use the standard single-spectrum fallback
        in the plotting layer.  Override to show residuals for multiple spectra
        simultaneously (e.g. both NuSTAR detectors).
        """
        return None

    def source_spec_info(
        self, spec_lookup: dict[int, dict]
    ) -> "dict[int, dict]":
        """Per-spec_id color and label overrides for source-kind spectra.

        Returns a dict mapping ``spec_id`` → ``{"color": str, "label": str}``.
        Any spec_id absent from the result falls back to ``KIND_STYLES["source"]``
        defaults.  Return an empty dict (default) to use the defaults for all.
        """
        return {}

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    @abstractmethod
    def render_components(
        self,
        ax,
        component_curves: dict,
        spec_lookup: dict[int, dict],
        modes: set[int],
        kind_styles: dict,
    ) -> None:
        """Draw model component curves onto the counts-panel axis.

        Parameters
        ----------
        ax
            Matplotlib ``Axes`` for the counts panel.
        component_curves
            Dict with keys ``"source"``, ``"astro_bkg"``, ``"inst_bkg"``,
            ``"other"`` — each a list of component dicts produced by
            ``XCMDataPlotter._collect_plot_arrays()``.
        spec_lookup
            ``{spec_id: spectrum_dict}`` — quick lookup from spectrum id to
            the spectrum entry, which carries ``"x"``, ``"y"``, ``"kind"``
            etc.
        modes
            Set of active mode numbers (subset of those declared in
            :attr:`modes`).
        kind_styles
            The shared :data:`~xip.plotting.KIND_STYLES` colour / linestyle
            dictionary.
        """
        ...
