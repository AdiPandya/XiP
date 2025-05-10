import matplotlib.pyplot as plt
import ipywidgets as widgets
from xspec import *
import numpy as np
import os
from xspec import AllData, AllModels, Spectrum, Xset, Model
from ipywidgets import FloatSlider, VBox, HBox, interactive_output
from IPython.display import display
import warnings
warnings.filterwarnings("ignore")

def get_free_parameters(AllModels, AllData):
    """
    Extract names and values of free (unfrozen, unlinked) parameters
    for all defined models in a multi-spectrum XSPEC analysis.

    Parameters:
        AllModels: XSPEC model object (e.g., AllModels)
        AllData: XSPEC data object (e.g., AllData)

    Returns:
        free_names: array of parameter names (free only)
        free_values: array of parameter values (free only)
        free_bounds: array of parameter bounds (free only)
        logmask: boolean array indicating logarithmic priors
    """
    Xset.chatter = 0
    Xset.logChatter = 0
    Models = [AllModels.sources[k] for k in sorted(AllModels.sources.keys())]
    N_spec = AllData.nSpectra

    free_name_model = np.array([], dtype=object)
    free_name_comp = np.array([], dtype=object)
    free_name_param = np.array([], dtype=object)
    free_values = np.array([], dtype=float)
    free_bounds = np.empty((0, 2), dtype=float)

    for n in range(1, N_spec + 1):
        for mod in Models:
            try:
                m = AllModels(n, mod)
            except:
                continue

            froz = np.array([m(i + 1).frozen for i in range(m.nParameters)])
            links = np.array([m(i + 1).link != '' for i in range(m.nParameters)])
            is_frozen_or_linked = froz | links

            pars = np.array([m(i + 1).values[0] for i in range(m.nParameters)])

            models = np.array([f"m_{mod}"] * m.nParameters, dtype=object)
            comp = np.array([comp for comp in m.componentNames for _ in getattr(m, comp).parameterNames], dtype=object)
            param = np.array([param for comp in m.componentNames for param in getattr(m, comp).parameterNames], dtype=object)

            free_name_model = np.concatenate((free_name_model, models[~is_frozen_or_linked]))
            free_name_comp = np.concatenate((free_name_comp, comp[~is_frozen_or_linked]))
            free_name_param = np.concatenate((free_name_param, param[~is_frozen_or_linked]))
            free_values = np.concatenate((free_values, pars[~is_frozen_or_linked]))

            bounds = np.zeros((m.nParameters, 2))
            for i in range(m.nParameters):
                if len(m(i + 1).values) > 1:
                    bounds[i] = [m(i + 1).values[2], m(i + 1).values[4]]
                else:
                    bounds[i] = [m(i + 1).values[0] - 1, m(i + 1).values[0] + 1]
            free_bounds = np.vstack((free_bounds, bounds[~is_frozen_or_linked]))

    priors = np.array(['lin' if ('PhoIndex' in n or 'nH' in n or 'factor' in n) else 'log' for n in free_name_param])
    logmask = (priors == 'log')

    return free_name_model, free_name_comp, free_name_param, free_values, free_bounds, logmask

def update_errorbar(errobj, x, y, xerr=None, yerr=None):
    ln, caps, bars = errobj
    # Ensure x, y, xerr, and yerr are numpy arrays
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    if xerr is not None and not isinstance(xerr, np.ndarray):
        xerr = np.array(xerr)
    if yerr is not None and not isinstance(yerr, np.ndarray):
        yerr = np.array(yerr)

    if len(bars) == 2:
        assert xerr is not None and yerr is not None, "Your errorbar object has 2 dimension of error bars defined. You must provide xerr and yerr."
        barsx, barsy = bars  # bars always exist (?)
        try:  # caps are optional
            errx_top, errx_bot, erry_top, erry_bot = caps
        except ValueError:  # in case there is no caps
            pass

    elif len(bars) == 1:
        assert (xerr is     None and yerr is not None) or\
               (xerr is not None and yerr is     None),  \
               "Your errorbar object has 1 dimension of error bars defined. You must provide xerr or yerr."

        if xerr is not None:
            barsx, = bars  # bars always exist (?)
            try:
                errx_top, errx_bot = caps
            except ValueError:  # in case there is no caps
                pass
        else:
            barsy, = bars  # bars always exist (?)
            try:
                erry_top, erry_bot = caps
            except ValueError:  # in case there is no caps
                pass

    ln.set_data(x,y)

    try:
        errx_top.set_xdata(x + xerr)
        errx_bot.set_xdata(x - xerr)
        errx_top.set_ydata(y)
        errx_bot.set_ydata(y)
    except NameError:
        pass
    try:
        barsx.set_segments([np.array([[xt, y], [xb, y]]) for xt, xb, y in zip(x + xerr, x - xerr, y)])
    except NameError:
        pass

    try:
        erry_top.set_xdata(x)
        erry_bot.set_xdata(x)
        erry_top.set_ydata(y + yerr)
        erry_bot.set_ydata(y - yerr)
    except NameError:
        pass
    try:
        barsy.set_segments([np.array([[x, yt], [x, yb]]) for x, yt, yb in zip(x, y + yerr, y - yerr)])
    except NameError:
        pass
    

class XSPECInteractivePlot:
    def __init__(self, Dir, bkgfile, rmf, arf, model_name, param_defaults=None, label_list=None, xspec_type = 'bkg',energy_range=(0.2, 8.0)):
        self.Dir = Dir
        self.bkgfile = bkgfile
        self.rmf = rmf
        self.arf = arf
        self.model_name = model_name
        self.param_defaults = param_defaults
        self.label_list = label_list
        self.xspec_type = xspec_type
        self.energy_range = energy_range
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.sliders = []
        self.component_lines = []
        self.PiB_line = None
        self.total_model_line = None
        self.res_points = None
        self.values = None
        self.model_type = None
        self.comp_names = None
        self.param_names = None
        self.bounds = None
        self.logmask = None
        self.m_src = None
        self.m_bkg = None
        self.m_fwc = None
        self.model_obj = None
        self.eng_bkg = None
        self.eng_bkg_err = None
        self.bkg_data = None
        self.bkg_data_err = None
        self.res_bkg = None
        self.res_bkg_err = None
        self.Pib_model = None
        self.total_model = None
        self.C = None
        self.dof = None

    def initialize_xspec(self):
        Xset.chatter = 0
        Xset.logChatter = 0
        Xset.xsect = 'vern'
        Xset.abund = 'wilm'
        Fit.statMethod = 'cstat'
        Xset.allowPrompting = False
        AllData.clear()
        AllModels.clear()
        pwd = os.getcwd()
        os.chdir(self.Dir)
        sp_bkg = Spectrum(self.bkgfile)
        sp_bkg.multiresponse[0] = self.rmf
        sp_bkg.multiresponse[0].arf = self.arf
        sp_bkg.multiresponse[1] = self.rmf
        os.chdir(pwd)
        AllData.ignore(f"**-{self.energy_range[0]},{self.energy_range[1]}-**")
    
    def set_model(self):
        Xset.chatter = 1
        Xset.logChatter = 0
        if self.xspec_type == 'bkg':
            self.m_bkg = Model(self.model_name, "bkg", 1)
            Xset.restore(f"{self.Dir}TM8_FWC_c010_mod_customized_bkg.dat")
            self.m_fwc = AllModels(1, "fwc")
            for _name in self.m_fwc.componentNames:
                if _name != "constant":
                    _comp = self.m_fwc.__getattribute__(_name)
                    for _pname in _comp.parameterNames:
                        _par = _comp.__getattribute__(_pname)
                        _par.frozen = True
        
        elif self.xspec_type == 'src':
            Xset.chatter = 1
            self.m_src = Model(self.model_name, "src", 1)
    
    def set_param_helper(self):
        self.initialize_xspec()
        self.set_model()
        self.model_type, self.comp_names, self.param_names, _, self.bounds, self.logmask = get_free_parameters(AllModels, AllData)
        print('Following parameters are free:')
        print('------------------------------------------------------------------------------------')
        print(f"{'No.':<8}{'Model':<15}{'Component':<20}{'Parameter':<20}{'Bounds':<30}")
        print('------------------------------------------------------------------------------------')
        
        for i, (model, comp, param, bound) in enumerate(zip(self.model_type, self.comp_names, self.param_names, self.bounds), start=1):
            print(f"{i:<8}{model:<15}{comp:<20}{param:<20}{str(bound):<30}")
        print('------------------------------------------------------------------------------------\n')
        print('Labels need to be set for following components:')
        unique_comps = np.unique(self.comp_names).tolist()
        multiplicative_comps = ['constant', 'TBabs', 'expfac']
        for comp in unique_comps:
            if not any(multi_comp in comp for multi_comp in multiplicative_comps):
                print(comp)
        
    def extract_parameters(self):
        if not self.param_defaults:
            raise ValueError("Argumnet param_defaults is required but not provided or are empty.")
        self.model_type, self.comp_names, self.param_names, _, self.bounds, self.logmask = get_free_parameters(AllModels, AllData)
        self.values = [param[0] if isinstance(param, tuple) else param for param in self.param_defaults]
        # new_free_mask = [not (isinstance(param, tuple) and param[1] == -1) for param in self.param_defaults]
        if len(self.param_names) != len(self.values):
            raise ValueError(f"Number of parameters ({len(self.param_names)}) does not match number of values ({len(self.values)})")
        if not self.label_list:
            unique_comps = np.unique(self.comp_names).tolist()
            multiplicative_comps = ['constant', 'TBabs', 'expfac']
            non_multiplicative_comps = []
            for comp in unique_comps:
                if not any(multi_comp in comp for multi_comp in multiplicative_comps):
                    non_multiplicative_comps.append(comp)
            self.label_list = non_multiplicative_comps

    def set_values(self, set_values):  
        for i in range(len(set_values)):
            set_values[i] = self.bounds[i][0] if set_values[i] < self.bounds[i][0] else set_values[i]
            set_values[i] = self.bounds[i][1] if set_values[i] > self.bounds[i][1] else set_values[i]
            new_frozen_mask = [isinstance(param, tuple) and param[1] == -1 for param in self.param_defaults]
            if self.model_type[i] == "m_src":
                self.model_obj = self.m_src
            elif self.model_type[i] == "m_bkg":
                self.model_obj = self.m_bkg
            elif self.model_type[i] == "m_fwc":
                self.model_obj = self.m_fwc
            try:
                component = getattr(self.model_obj, self.comp_names[i])
                setattr(component, self.param_names[i], set_values[i])
            except Exception as e:
                print(f"Error setting parameter {self.model_type[i]}.{self.comp_names[i]}.{self.param_names[i]}: {e}")
            if new_frozen_mask[i]:
                component = getattr(self.model_obj, self.comp_names[i])
                # setattr(component, self.param_names[i], set_values[i])
                component.__getattribute__(self.param_names[i]).frozen = True

    def perform_fit(self):
        self.set_values(self.values)
        # Xset.chatter = 10
        # print(AllModels.show())
        Fit.query = "yes"
        Fit.nIterations = 1000
        Fit.perform()
        # Xset.chatter = 0
        self.model_type, self.comp_names, self.param_names, self.values, self.bounds, self.logmask= get_free_parameters(AllModels, AllData)
        # print(self.values), print(len(self.model))
        self.set_values(self.values)
        self.C = Fit.statistic
        self.dof = Fit.dof

    def plot_data(self):
        Plot.device = "/null"
        Plot.add = True
        Plot.xAxis = "keV"
        Plot.setRebin(minSig=20, maxBins=100)
        Plot('ldata delchi')
        self.bkg_data = Plot.y(1)
        self.bkg_data_err = Plot.yErr(1)
        self.eng_bkg = Plot.x(1)
        self.eng_bkg_err = Plot.xErr(1)
        self.total_model = Plot.model(1)
        self.res_bkg = Plot.y(1, 2)
        self.res_bkg_err = Plot.yErr(1, 2)
        if self.xspec_type == 'bkg':
            self.Pib_model = np.zeros(len(Plot.model(1)))
            for i in range(len(self.comp_names), Plot.nAddComps() - 1):
                self.Pib_model += np.array(Plot.addComp(i, 1))

    def create_plot(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0})
        self.ax1.errorbar(self.eng_bkg, self.bkg_data, xerr=self.eng_bkg_err, yerr=self.bkg_data_err, fmt='.', color="black", label="bkg data", alpha=0.5, lw=1.2)
        self.total_model_line, = self.ax1.plot(self.eng_bkg, self.total_model, label="Full model")
        for i in range(len(self.label_list)):
            line, = self.ax1.plot(self.eng_bkg, Plot.addComp(i+1, 1), label=self.label_list[i], linestyle="--")
            self.component_lines.append(line)
        if self.xspec_type == 'bkg':
            self.PiB_line, = self.ax1.plot(self.eng_bkg, self.Pib_model, label="PIB model", linestyle="--")
        self.ax1.set_yscale("log")
        self.ax1.set_xscale("log")
        self.ax1.set_ylabel(r"$\mathrm{Counts\ s^{-1}\ keV^{-1}}$")
        self.ax1.set_ylim(3, 2e2)
        self.ax1.legend()
        self.res_points = self.ax2.errorbar(self.eng_bkg, self.res_bkg, xerr=self.eng_bkg_err, yerr=self.res_bkg_err, fmt='.', color="black", label="residuals", alpha=0.5, lw=1.2)
        self.ax2.axhline(0, color='gray', linestyle='--')
        self.ax2.set_xscale("log")
        self.ax2.set_xlabel("Energy [keV]")
        self.ax2.set_ylabel("Residuals")
        self.ax2.set_xticks([0.2, 0.5, 1, 2, 5])
        self.ax2.set_xticklabels(['0.2', '0.5', '1', '2', '5'])
        self.ax2.set_ylim(-5, 5)
        self.ax2.set_yticks([-3, 0, 3])
        self.ax2.axhspan(-3, 3, color='gray', alpha=0.15)
        self.ax1.set_title(f"CSTAT/d.o.f. = {(self.C / self.dof):.2f} ({self.C:.2f}/{self.dof})")

    def update_model(self, new_values):
        self.set_values(new_values)
        self.C = Fit.statistic
        self.dof = Fit.dof
        Plot('ldata delchi')
        self.total_model = Plot.model(1)
        self.res_bkg = Plot.y(1, 2)
        self.res_bkg_err = Plot.yErr(1, 2)
        if self.xspec_type == 'bkg':
            self.Pib_model = np.zeros(len(Plot.model(1)))
            for i in range(Plot.nAddComps() - 26, Plot.nAddComps() - 1):
                self.Pib_model += np.array(Plot.addComp(i, 1))
            self.PiB_line.set_ydata(self.Pib_model)
        
        self.total_model_line.set_ydata(self.total_model)
        for i in range(len(self.label_list)):
            self.component_lines[i].set_ydata(Plot.addComp(i + 1, 1))
            
        self.ax1.set_title(f"CSTAT/d.o.f. = {(self.C / self.dof):.2f} ({self.C:.2f}/{self.dof})")
        update_errorbar(self.res_points, self.eng_bkg, self.res_bkg, xerr=self.eng_bkg_err, yerr=self.res_bkg_err)
        self.fig.canvas.draw_idle()

    def create_sliders(self):
        self.sliders = [
            FloatSlider(
                min=val - abs(val*0.9),
                max=val + abs(val*10),
                value=val,
                step=val / 10,
                continuous_update=True,
                description=f'{comp}.{param}:',  # Add a colon for better spacing
                style={'description_width': '150px'},  # Adjust description width for more space
                readout_format='.2f',
                layout=widgets.Layout(width='400px')  # Make sliders bigger
            )
            for comp, param, val in zip(self.comp_names, self.param_names, self.values)
        ]

    def create_interactive_layout(self):
        def proxy_function(**kwargs):
            ordered_values = [kwargs[k] for k in sorted(kwargs.keys())]
            return self.update_model(ordered_values)

        controls = {f'v{i}': slider for i, slider in enumerate(self.sliders)}
        output = interactive_output(proxy_function, controls)
        reset_button = widgets.Button(description="Reset", button_style="info")

        def reset_parameters(_):
            for slider, default_value in zip(self.sliders, self.values):
                slider.value = default_value
            self.update_model(self.values)

        reset_button.on_click(reset_parameters)
        layout = VBox([HBox([VBox(self.sliders), reset_button]), output])
        display(layout)

    def run(self):
        self.initialize_xspec()
        self.set_model()
        self.extract_parameters()
        self.perform_fit()
        self.plot_data()
        self.create_plot()
        self.create_sliders()
        self.create_interactive_layout()