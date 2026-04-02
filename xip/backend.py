import argparse
import json
import matplotlib.pyplot as plt
from xspec import *
import numpy as np
import os
import re
import tempfile
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

plt.style.use("default")
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)
plt.rc('axes', linewidth=1.15)

size = 11
plt.rc("mathtext", fontset="dejavuserif")
plt.rc('font', family='DejaVu Serif', size=size)

# Load multiplicative model list from database.
_DB_PATH = Path(__file__).parent / "xspec_model_db.json"
with _DB_PATH.open() as _f:
    _DB = json.load(_f)
_MULT_COMPONENTS: frozenset[str] = frozenset(_DB.get("multiplicative_models", []))

# KIND_STYLES is defined in plotting.py and imported here for backward compatibility.
from xip.plotting import KIND_STYLES  # noqa: E402


class XCMDataPlotter:
    def __init__(self, xcm_file, energy_range=(0.2, 8.0), min_sig=20, max_bins=100, perform_fit=True):
        self.xcm_path = Path(xcm_file).expanduser().resolve()
        self.energy_range = energy_range
        self.minSig = min_sig
        self.maxBins = max_bins
        self.perform_fit = perform_fit
        self.spectrum_kind_map = {}
        self.model_kind_map = {}
        self.model_order = []
        self.model_component_labels = {}
        self.component_label_counts = {}
        self._multiplicative_components = _MULT_COMPONENTS
        self.fig = None
        self.ax_counts = None
        self.ax_residuals = None
        self.statistic = None
        self.dof = None
        self._cached_comp_counts = None
        # Populated by _assign_spectrum_kinds / _assign_model_kinds after session load
        self._kind_warnings: dict[int, str] = {}
        self._model_kind_warnings: list[str] = []

    def _configure_xspec(self):
        Xset.chatter = 0
        Xset.xsect = 'vern'
        Xset.abund = 'wilm'
        Fit.statMethod = 'cstat'
        Xset.allowPrompting = False
        AllData.clear()
        AllModels.clear()

    def _restore_session(self):
        if not self.xcm_path.exists():
            raise FileNotFoundError(f"Unable to find XCM file at {self.xcm_path}")

        # Pre-process: resolve all `cd` commands to absolute paths so the XCM
        # works regardless of where it lives on disk.  Write to a temp file and
        # restore from that so the original file is never modified.
        xcm_dir = self.xcm_path.parent
        processed_lines = []
        current_dir = xcm_dir
        _cd_re = re.compile(r'^(cd)\s+(.+)$', re.IGNORECASE)
        for line in self.xcm_path.read_text().splitlines():
            m = _cd_re.match(line.rstrip())
            if m:
                target = Path(m.group(2).strip())
                if not target.is_absolute():
                    target = (current_dir / target).resolve()
                current_dir = target
                processed_lines.append(f"cd {target}")
            else:
                processed_lines.append(line)
        processed_text = "\n".join(processed_lines)

        cwd = os.getcwd()
        tmp_path = None
        try:
            # Write the processed XCM to a temp file in the same directory so
            # any relative data paths not guarded by 'cd' also resolve correctly.
            fd, tmp_path = tempfile.mkstemp(suffix=".xcm", dir=xcm_dir)
            with os.fdopen(fd, 'w') as fh:
                fh.write(processed_text)
            os.chdir(xcm_dir)
            Xset.restore(Path(tmp_path).name)
        finally:
            os.chdir(cwd)
            if tmp_path and Path(tmp_path).exists():
                Path(tmp_path).unlink()
        if AllData.nSpectra == 0:
            raise RuntimeError("The provided XCM file did not load any spectra.")
        if self.energy_range is not None:
            emin, emax = self.energy_range
            AllData.ignore(f"**-{emin},{emax}-**")
        if self.perform_fit:
            Fit.query = "yes"
            Fit.nIterations = 1000
            Fit.perform()
        self.statistic = Fit.statistic
        self.dof = Fit.dof
        self._assign_spectrum_kinds()
        self._assign_model_kinds()

    def _assign_spectrum_kinds(self):
        """Map the loaded spectra to source/background slots.

        Uses filename keywords first (supports eROSITA, NuSTAR, XMM, Chandra,
        Suzaku naming conventions); falls back to load order and records a
        warning in self._kind_warnings when it does.
        """
        _BKG_KEYWORDS = {
            # Generic
            "bkg", "background", "back", "bgd", "bgr",
            # NuSTAR
            "bk",
            # XMM-Newton pn/MOS particle background
            "pm", "oot",
            # Suzaku / XMM particle background
            "nxb", "qpb",
            # Filter-wheel-closed / FWC
            "fo", "fwc",
        }
        _SRC_KEYWORDS = {
            # Generic
            "src", "source", "signal",
            # NuSTAR
            "sr",
            # NuSTAR FPMA/FPMB (rebinned / grouped files like fpma_optmin20.pha)
            "fpma", "fpmb",
            # Chandra / XMM
            "obj", "target",
            # Grouped source
            "grp",
        }
        preferred_order = ["source", "bkg"]
        self.spectrum_kind_map = {}
        self._kind_warnings = {}
        for idx, spec_id in enumerate(range(1, AllData.nSpectra + 1)):
            kind = None
            try:
                fname = (AllData(spec_id).fileName or "").lower()
                stem = Path(fname).stem
                # Split on underscores, hyphens, spaces, and dots
                parts = set(re.split(r"[_\-\s\.]+", stem))
                if parts & _BKG_KEYWORDS:
                    kind = "bkg"
                elif parts & _SRC_KEYWORDS:
                    kind = "source"
            except Exception:
                pass
            if kind is None:
                kind = preferred_order[idx] if idx < len(preferred_order) else "other"
                self._kind_warnings[spec_id] = (
                    f"Spectrum {spec_id}: kind inferred from load order as \"{kind}\" "
                    "(filename had no recognised source/background keyword)"
                )
            self.spectrum_kind_map[spec_id] = kind

    def _assign_model_kinds(self):
        """Map XSPEC source numbers to conceptual model kinds in load order.

        Layout rules:
          1 model  → source
          2 models → source, astro_bkg (use Default profile for bkg view)
          3 models → source, astro_bkg, inst_bkg  (eROSITA layout)
          >3       → first 3 as eROSITA, remainder as "other"
        """
        self.model_kind_map = {}
        self.model_order = []
        self._model_kind_warnings = []
        sources = getattr(AllModels, "sources", {})
        if not sources:
            return
        n = len(sources)
        if n == 1:
            kind_priority = ["source"]
        elif n == 2:
            kind_priority = ["source", "astro_bkg"]
            self._model_kind_warnings.append(
                "2 models loaded — mapped to source (model 1) and background (model 2). "
                "Use the 'Default' profile with modes 1 and 3 to view each."
            )
        elif n == 3:
            kind_priority = ["source", "astro_bkg", "inst_bkg"]
            self._model_kind_warnings.append(
                "3 models loaded — assumed eROSITA layout "
                "(source / astro_bkg / inst_bkg). "
                "Switch to 'Default' profile if this is incorrect."
            )
        else:
            kind_priority = ["source", "astro_bkg", "inst_bkg"]
            self._model_kind_warnings.append(
                f"{n} models loaded — first 3 mapped to eROSITA layout, "
                f"{n - 3} additional model(s) mapped to 'other'."
            )
        for idx, (src_no, name) in enumerate(sorted(sources.items())):
            kind = kind_priority[idx] if idx < len(kind_priority) else "other"
            self.model_kind_map[name] = kind
            self.model_order.append(name)
        self._capture_component_labels()

    def _get_model_object(self, model_name):
        data_groups = set([1])
        for spec_id in range(1, AllData.nSpectra + 1):
            try:
                data_groups.add(AllData(spec_id).dataGroup)
            except Exception:
                continue
        for group in sorted(data_groups):
            try:
                return AllModels(group, model_name)
            except Exception:
                continue
        return None

    def _capture_component_labels(self):
        self.model_component_labels = {}
        if not self.model_order:
            return
        for name in self.model_order:
            model_obj = self._get_model_object(name)
            if model_obj is None:
                continue
            expression = getattr(model_obj, "expression", "") or ""
            additive_names = self._extract_additive_component_names(expression)
            if not additive_names:
                additive_names = list(getattr(model_obj, "componentNames", []))
            filtered_names = [
                comp_name
                for comp_name in additive_names
                if comp_name.lower() not in self._multiplicative_components
            ]
            if not filtered_names:
                continue
            deduped = []
            seen = {}
            for comp_name in filtered_names:
                count = seen.get(comp_name, 0) + 1
                seen[comp_name] = count
                deduped.append(f"{comp_name} ({count})" if count > 1 else comp_name)
            self.model_component_labels[name] = deduped
        self.component_label_counts = {
            i: len(model_labels)
            for i, model_labels in enumerate(self.model_component_labels.values(), start=1)
        }


    @staticmethod
    def _extract_additive_component_names(expression):
        if not expression:
            return []
        cleaned = expression.replace(" ", "")
        pattern = re.compile(r"([A-Za-z0-9_]+)<\d+>")
        names = []
        for match in pattern.finditer(cleaned):
            end = match.end()
            next_char = cleaned[end:end + 1]
            if next_char == '(':
                continue
            names.append(match.group(1))
        return names

    def _spectrum_kind(self, spec_id):
        return self.spectrum_kind_map.get(spec_id, "other")

    def _describe_spectrum(self, spec_id):
        try:
            spec = AllData(spec_id)
            if spec.fileName:
                return Path(spec.fileName).name
        except Exception:
            pass
        kind = self._spectrum_kind(spec_id)
        return f"{kind.replace('_', ' ').title()} ({spec_id})"

    def _collect_plot_arrays(self, recompute_structure: bool = True):
        Plot.device = "/null"
        Plot.add = True
        Plot.xAxis = "keV"
        Plot.setRebin(minSig=self.minSig, maxBins=self.maxBins)
        Plot('ldata delchi')

        spectra = {
            "source": None,
            "bkg": None,
            "others": []
        }
        baseline_counts = {}
        component_arrays = {}
            
        for spec_id in range(1, AllData.nSpectra + 1):
            kind = self._spectrum_kind(spec_id)
            model_curve = np.array(Plot.model(spec_id))
            
            entry = {
                "id": spec_id,
                "kind": kind,
                "label": self._describe_spectrum(spec_id),
                "x": np.array(Plot.x(spec_id)),
                "xerr": np.array(Plot.xErr(spec_id)),
                "y": np.array(Plot.y(spec_id)),
                "yerr": np.array(Plot.yErr(spec_id)),
                "model": model_curve,
                # "components": components,
            }

            if kind in ("source", "bkg") and spectra[kind] is None:
                # First spectrum of this kind occupies the primary slot.
                spectra[kind] = entry
            else:
                # Additional source/bkg spectra (e.g. FPMB) go into 'others'
                # with their original kind preserved so profiles can find them.
                spectra["others"].append(entry)
            total_components = Plot.nAddComps(spec_id)
            
            component_arrays[spec_id] = [
                np.array(Plot.addComp(comp_idx + 1, spec_id))
                for comp_idx in range(total_components)
            ]
            if self.component_label_counts.get(spec_id, 0) == 1:
                total_components += 1
                sum_curve = np.sum(component_arrays[spec_id], axis=0)
                src_component = Plot.model(spec_id) - sum_curve
                component_arrays[spec_id].insert(0, src_component)
            
            baseline_counts[spec_id] = total_components

        residuals_by_spec: dict[int, dict] = {}
        for spec_id in range(1, AllData.nSpectra + 1):
            try:
                residuals_by_spec[spec_id] = {
                    "x":    np.array(Plot.x(spec_id, 2)),
                    "xerr": np.array(Plot.xErr(spec_id, 2)),
                    "y":    np.array(Plot.y(spec_id, 2)),
                    "yerr": np.array(Plot.yErr(spec_id, 2)),
                    "label": f"Residuals ({self._describe_spectrum(spec_id)})",
                }
            except Exception:
                pass
        component_curves = {"source": [], "astro_bkg": [], "inst_bkg": [], "other": []}
        if any(baseline_counts.values()) and self.model_order:
            if recompute_structure or self._cached_comp_counts is None:
                comp_counts = self._calculate_model_component_counts(baseline_counts)
                self._cached_comp_counts = comp_counts
            else:
                comp_counts = self._cached_comp_counts
            component_curves = self._assemble_component_curves(component_arrays, comp_counts)
        return spectra, residuals_by_spec, component_curves

    def _calculate_model_component_counts(self, baseline_counts):
        counts = {
            model_name: {spec_id: 0 for spec_id in baseline_counts}
            for model_name in self.model_order
        }
        if not counts:
            return counts
        if len(self.model_order) == 1:
            sole = self.model_order[0]
            counts[sole] = dict(baseline_counts)
            return counts

        # Strategy: deactivate models 0..N-2 one at a time and measure the
        # drop in nAddComps per spectrum.  The LAST model is never deactivated;
        # its counts are computed as (baseline − sum of all earlier models).
        #
        # This avoids XSPEC's C-level SIGSEGV that occurs when ANY spectrum is
        # left with zero active models after setInactive.  In every standard
        # layout (2-model src+bkg, 3-model eROSITA, …) the last model is the
        # one that covers the background spectrum exclusively, so skipping its
        # deactivation is always safe.
        n_models = len(self.model_order)

        for i, model_name in enumerate(self.model_order[:-1]):
            try:
                AllModels.setInactive(model_name)
            except Exception:
                continue
            Plot('ldata delchi')
            for spec_id in baseline_counts:
                base_total = baseline_counts[spec_id]
                # baseline_counts was inflated by +1 for spectra whose first
                # model has a single additive component (single-component hack).
                # When deactivating any model other than the first, that
                # artificial +1 is still in baseline but XSPEC returns 0 for
                # nAddComps on those spectra, so we must subtract it back out.
                if i > 0 and self.component_label_counts.get(spec_id, 0) == 1:
                    base_total -= 1
                try:
                    current_total = Plot.nAddComps(spec_id)
                except Exception:
                    current_total = 0
                counts[model_name][spec_id] = max(0, base_total - current_total)
            try:
                AllModels.setActive(model_name)
            except Exception:
                pass
            Plot('ldata delchi')

        # Last model: assign whatever baseline is left after all others.
        last_model = self.model_order[-1]
        for spec_id in baseline_counts:
            already = sum(counts[m][spec_id] for m in self.model_order[:-1])
            counts[last_model][spec_id] = max(0, baseline_counts[spec_id] - already)

        return counts

    def _assemble_component_curves(self, component_arrays, component_counts):
        component_curves = {"source": [], "astro_bkg": [], "inst_bkg": [], "other": []}
        model_offsets = {name: 0 for name in self.model_order}
        for spec_id, comp_list in component_arrays.items():
            offset = 0
            for model_name in self.model_order:
                model_counts = component_counts.get(model_name, {})
                count = model_counts.get(spec_id, 0)
                if count <= 0:
                    continue
                slice_end = offset + count
                comps = comp_list[offset:slice_end]
                offset = slice_end
                if not comps:
                    continue
                kind = self.model_kind_map.get(model_name, "other")
                label_offset = model_offsets.get(model_name, 0)
                labels = self.model_component_labels.get(model_name, [])
                for idx, curve in enumerate(comps, start=1):
                    label_idx = label_offset + idx - 1
                    comp_label = labels[label_idx % len(labels)] if labels else f"Component {idx}"
                    component_curves[kind].append(
                        {
                            "spec_id": spec_id,
                            "curve": curve,
                            "model_name": model_name,
                            "component_index": idx,
                            "component_label": comp_label,
                        }
                    )
                model_offsets[model_name] = label_offset + len(comps)
        return component_curves

    def _plot(self):
        from xip.plotting import render_xspec_plot
        spectra, residuals_by_spec, component_curves = self._collect_plot_arrays()
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(9, 5), sharex=True,
            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0},
        )
        render_xspec_plot(
            ax1, ax2,
            spectra, residuals_by_spec, component_curves,
            self.statistic, self.dof,
            energy_range=self.energy_range,
        )
        fig.tight_layout()
        plt.show()

    def run(self):
        print(f"Loading XSPEC session from {self.xcm_path}\n")
        self._configure_xspec()
        self._restore_session()
        print(self._build_session_summary())
        self._plot()
        plt.show()


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Plot XSPEC data+model directly from an XCM session file.")
    parser.add_argument("xcm_file", help="Path to the XSPEC .xcm session file")
    parser.add_argument(
        "--energy-range",
        nargs=2,
        type=float,
        metavar=("E_MIN", "E_MAX"),
        default=(0.2, 8.0),
        help="Energy range in keV to retain before plotting (default: 0.2 8.0)"
    )
    parser.add_argument(
        "--no-energy-filter",
        action="store_true",
        help="Skip ignoring data outside the supplied energy range"
    )
    parser.add_argument("--min-sig", type=int, default=20, help="Minimum sigma for XSPEC rebinning (default: 20)")
    parser.add_argument("--max-bins", type=int, default=100, help="Maximum bins for XSPEC rebinning (default: 100)")
    parser.add_argument("--no-fit", action="store_true", help="Do not re-fit after loading the session")
    return parser.parse_args()


def main():
    args = parse_cli_args()
    energy_range = None if args.no_energy_filter else tuple(args.energy_range)
    plotter = XCMDataPlotter(
        xcm_file=args.xcm_file,
        energy_range=energy_range,
        min_sig=args.min_sig,
        max_bins=args.max_bins,
        perform_fit=not args.no_fit
    )
    plotter.run()


if __name__ == "__main__":
    main()