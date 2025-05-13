import matplotlib.pyplot as plt
import ipywidgets as widgets
from xspec import *
import numpy as np
import os
from xspec import AllData, AllModels, Spectrum, Xset, Model
from ipywidgets import FloatSlider, VBox, HBox, interactive_output
from IPython.display import display
import warnings
from utils import get_free_parameters, update_errorbar

warnings.filterwarnings("ignore")

class XSPECInteractivePlot:
    def __init__(self, data_config, model_config, xspec_type='src', energy_range=(0.2, 8.0)):
        # Data configuration
        self.Dir = data_config.get('Dir')
        self.srcfile = data_config.get('src_file')
        self.bkgfile = data_config.get('bkg_file')
        self.src_rmf = data_config.get('src_rmf')
        self.src_arf = data_config.get('src_arf')
        self.bkg_rmf = data_config.get('bkg_rmf')
        self.bkg_arf = data_config.get('bkg_arf')
        self.fwc_data = data_config.get('fwc_data')

        # Model configuration
        self.model_name = model_config.get('model_name')
        self.param_defaults = model_config.get('param_defaults', [])
        self.label_list = model_config.get('label_list', [])

        # Other settings
        self.xspec_type = xspec_type
        self.energy_range = energy_range
        self.chatter = 0
        self.perform_fit = True
        self.show_bkg_data = True
        self.minSig = 20
        self.maxBins = 100

        # Internal attributes
        self.AllData = None
        self.AllModels = None
        
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.sliders = []
        self.component_lines = []
        self.PiB_line = None
        self.total_model_line = None
        self.total_src_line = None
        self.total_bkg_line = None
        self.res_points = None
        self.values = None
        self.model_type = None
        self.comp_names = None
        self.param_names = None
        self.filtered_model_comp_names = None
        self.bkg_model_comp_len = None
        self.bounds = None
        self.logmask = None
        self.m_src = None
        self.m_bkg = None
        self.m_fwc = None
        self.model_obj = None
        self.plotdata = {
            "eng_bkg": None,
            "eng_bkg_err": None,
            "bkg_data": None,
            "bkg_data_err": None,
            "eng_src": None,
            "eng_src_err": None,
            "src_data": None,
            "src_data_err": None,
        }
        self.residuals = None
        self.residuals_err = None
        self.Pib_model = None
        self.total_model = None
        self.total_src_model = None
        self.total_bkg_model = None
        self.C = None
        self.dof = None

    def initialize_xspec(self):
        Xset.chatter = self.chatter
        Xset.xsect = 'vern'
        Xset.abund = 'wilm'
        Fit.statMethod = 'cstat'
        Xset.allowPrompting = False
        self.AllData = AllData
        self.AllModels = AllModels
        self.AllData.clear()
        self.AllModels.clear()
        pwd = os.getcwd()
        os.chdir(self.Dir)
        if self.xspec_type == 'bkg':
            sp_bkg = Spectrum(self.bkgfile)
            if not (self.bkg_rmf and self.bkg_arf):
                self.bkg_rmf = sp_bkg.response.rmf
                self.bkg_arf = sp_bkg.response.arf
            sp_bkg.multiresponse[0] = self.bkg_rmf
            sp_bkg.multiresponse[0].arf = self.bkg_arf
            sp_bkg.multiresponse[1] = self.bkg_rmf
        elif self.xspec_type == 'src':
            sp_src = Spectrum(self.srcfile)
            if self.src_rmf and self.src_arf:
                sp_src.response.rmf = self.src_rmf
                sp_src.response.arf = self.src_arf
        elif self.xspec_type == 'src_bkg':
            sp_src = Spectrum(self.srcfile)
            if not (self.src_rmf and self.src_arf):
                self.src_rmf = sp_src.response.rmf
                self.src_arf = sp_src.response.arf
            sp_src.multiresponse[0].arf = self.src_arf
            sp_src.multiresponse[1] = self.src_rmf
            sp_src.multiresponse[1].arf = self.src_arf
            sp_src.multiresponse[2] = self.src_rmf

            sp_bkg = Spectrum(self.bkgfile)
            if not (self.bkg_rmf and self.bkg_arf):
                self.bkg_rmf = sp_bkg.response.rmf
                self.bkg_arf = sp_bkg.response.arf
            sp_bkg.multiresponse[0] = None
            sp_bkg.multiresponse[1] = self.bkg_rmf
            sp_bkg.multiresponse[1].arf = self.bkg_arf
            sp_bkg.multiresponse[2] = self.bkg_rmf
        os.chdir(pwd)
        self.AllData.ignore(f"**-{self.energy_range[0]},{self.energy_range[1]}-**")
    
    def set_model(self):
        self.initialize_xspec()
        if self.xspec_type == 'bkg':
            self.m_bkg = Model(self.model_name, "bkg", 1)
            try:
                Xset.restore(f"{self.Dir}/{self.fwc_data}")
                self.m_fwc = self.AllModels(1, "fwc")
                for _name in self.m_fwc.componentNames:
                    if _name != "constant":
                        _comp = self.m_fwc.__getattribute__(_name)
                        for _pname in _comp.parameterNames:
                            _par = _comp.__getattribute__(_pname)
                            _par.frozen = True
            except:
                print("No FWC model found, proceeding without it.")
                pass

        elif self.xspec_type == 'src':
            self.m_src = Model(self.model_name, "src", 1)
            
        elif self.xspec_type == 'src_bkg':
            if not '|' in self.model_name:
                raise ValueError("Model name must contain '|' to separate source and background models.")
            src_model, bkg_model = self.model_name.split('|', 1)
            self.m_src = Model(src_model, "src", 1)
            self.m_bkg = Model(bkg_model, "bkg", 2)
            try:
                Xset.restore(f"{self.Dir}/{self.fwc_data}")
                self.m_fwc = self.AllModels(1, "fwc")
                for _name in self.m_fwc.componentNames:
                    if _name != "constant":
                        _comp = self.m_fwc.__getattribute__(_name)
                        for _pname in _comp.parameterNames:
                            _par = _comp.__getattribute__(_pname)
                            _par.frozen = True
            except:
                print("No FWC model found, proceeding without it.")
                pass
        
    def set_param_helper(self):
        self.initialize_xspec()
        self.set_model()
        self.model_type, self.comp_names, self.param_names, _, self.bounds, self.logmask = get_free_parameters(AllModels, self.AllData)
        print('Following parameters are free:')
        print('------------------------------------------------------------------------------------')
        print(f"{'No.':<8}{'Model':<15}{'Component':<20}{'Parameter':<20}{'Bounds':<30}")
        print('------------------------------------------------------------------------------------')
        
        for i, (model, comp, param, bound) in enumerate(zip(self.model_type, self.comp_names, self.param_names, self.bounds), start=1):
            print(f"{i:<8}{model:<15}{comp:<20}{param:<20}{str(bound):<30}")
        print('------------------------------------------------------------------------------------\n')
        print('Labels need to be set for following components:')
        
        model_comp_names = [f"{model}.{comp}" for model, comp in zip(self.model_type, self.comp_names)]
        unique_comps_mask = [name not in seen and not seen.add(name) for seen in [set()] for name in model_comp_names]
        multiplicative_comps = ['constant', 'TBabs', 'expfac']
        additive_comp_mask = [not any(multi_comp in model_comp_names[i].split('.')[-1] for multi_comp in multiplicative_comps) for i in range(len(model_comp_names))]
        combined_mask = np.logical_and(unique_comps_mask, additive_comp_mask)
        self.filtered_model_comp_names = np.array(model_comp_names)[combined_mask].tolist()
        for labels in self.filtered_model_comp_names:
            print(labels)
        print('\n------------------------------------------------------------------------------------')
        
    def extract_parameters(self):
        if not self.param_defaults:
            raise ValueError("Argumnet param_defaults is required but not provided or are empty.")
        self.model_type, self.comp_names, self.param_names, _, self.bounds, self.logmask = get_free_parameters(self.AllModels, self.AllData)
        self.values = [param[0] if isinstance(param, tuple) else param for param in self.param_defaults]
        
        if len(self.param_names) != len(self.values):
            raise ValueError(f"Number of model parameters ({len(self.param_names)}) does not match number of input values ({len(self.values)})")

        free_mask = [not (isinstance(f_mask, tuple) and f_mask[1] == -1) for f_mask in self.param_defaults]
        model_comp_names = [f"{model}.{comp}" for model, comp in zip(self.model_type, self.comp_names)]
        free_m_comp_names = np.array(model_comp_names)[free_mask]
        unique_comps_mask = [name not in seen and not seen.add(name) for seen in [set()] for name in free_m_comp_names]
        unique_m_comp_names = np.array(free_m_comp_names)[unique_comps_mask]
        multiplicative_comps = ['constant', 'TBabs', 'expfac']
        additive_comp_mask = [not any(multi_comp in unique_m_comp_names[i].split('.')[-1] for multi_comp in multiplicative_comps) for i in range(len(unique_m_comp_names))]
        self.filtered_model_comp_names = np.array(unique_m_comp_names)[additive_comp_mask]  
        
        if not self.label_list or len(self.label_list) != len(self.filtered_model_comp_names):
            self.label_list = self.filtered_model_comp_names

    def set_values(self, set_values, frozen_mask=True):  
        if frozen_mask:
            new_frozen_mask = [isinstance(param, tuple) and param[1] == -1 for param in self.param_defaults]
        else:
            new_frozen_mask = [False] * len(self.param_names)
        for i in range(len(set_values)):
            set_values[i] = self.bounds[i][0] if set_values[i] < self.bounds[i][0] else set_values[i]
            set_values[i] = self.bounds[i][1] if set_values[i] > self.bounds[i][1] else set_values[i]
            
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

    def final_set(self):
        Xset.chatter = self.chatter
        self.set_values(self.values)
        if self.perform_fit:
            Fit.query = "yes"
            Fit.nIterations = 1000
            Fit.perform()
        self.model_type, self.comp_names, self.param_names, self.values, self.bounds, self.logmask= get_free_parameters(self.AllModels, self.AllData)
        self.set_values(self.values, frozen_mask=False)
        self.C = Fit.statistic
        self.dof = Fit.dof
        
    def plot_data(self):
        Plot.device = "/null"
        Plot.add = True
        Plot.xAxis = "keV"
        Plot.setRebin(minSig=self.minSig, maxBins=self.maxBins)
        Plot('ldata delchi')
        if self.xspec_type == 'src_bkg':
            self.plotdata["eng_src"] = Plot.x(1)
            self.plotdata["eng_src_err"] = Plot.xErr(1)
            self.plotdata["src_data"] = Plot.y(1)
            self.plotdata["src_data_err"] = Plot.yErr(1)
            self.plotdata["eng_bkg"] = Plot.x(2)
            self.plotdata["eng_bkg_err"] = Plot.xErr(2)
            self.plotdata["bkg_data"] = Plot.y(2)
            self.plotdata["bkg_data_err"] = Plot.yErr(2)
            self.total_model = Plot.model(1)
            self.total_src_model = np.zeros(len(Plot.model(1)))
            if Plot.nAddComps(2) < Plot.nAddComps(1):
                for i in range(1, Plot.nAddComps(1)-Plot.nAddComps(2) + 1):
                    self.total_src_model += np.array(Plot.addComp(i, 1))
                self.total_bkg_model =Plot.model(2)
            if Plot.nAddComps(2) == Plot.nAddComps(1):
                self.total_bkg_model = np.zeros(len(Plot.model(1)))
                for i in range(1, Plot.nAddComps()):
                    self.total_bkg_model += np.array(Plot.addComp(i, 1))
                self.total_src_model = self.total_model - self.total_bkg_model
                self.total_bkg_model = Plot.model(2)
                
        elif self.xspec_type == 'src':
            self.plotdata["eng_src"] = Plot.x(1)
            self.plotdata["eng_src_err"] = Plot.xErr(1)
            self.plotdata["src_data"] = Plot.y(1)
            self.plotdata["src_data_err"] = Plot.yErr(1)
            self.total_model = Plot.model(1)

        elif self.xspec_type == 'bkg':
            self.plotdata["eng_bkg"] = Plot.x(1)
            self.plotdata["eng_bkg_err"] = Plot.xErr(1)
            self.plotdata["bkg_data"] = Plot.y(1)
            self.plotdata["bkg_data_err"] = Plot.yErr(1)
            self.total_model = Plot.model(1)
            
        self.residuals = Plot.y(1, 2)
        self.residuals_err = Plot.yErr(1, 2)
        if self.xspec_type == 'bkg':
            self.Pib_model = np.zeros(len(Plot.model(1)))
            bkg_filtered_model_comp_names = [comp for comp in self.filtered_model_comp_names if comp.split('.')[0] == 'm_bkg']
            self.bkg_model_comp_len = len(bkg_filtered_model_comp_names)
            for i in range(self.bkg_model_comp_len+1, Plot.nAddComps() + 1):
                self.Pib_model += np.array(Plot.addComp(i, 1))
            
    def create_plot(self):
        self.plot_data()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0})
        if self.xspec_type == 'src_bkg':
            self.ax1.errorbar(self.plotdata["eng_src"], self.plotdata["src_data"], xerr=self.plotdata["eng_src_err"], yerr=self.plotdata["src_data_err"], fmt='.', color="black", label="src data", alpha=0.5, lw=1.2)
            if self.show_bkg_data:
                self.ax1.errorbar(self.plotdata["eng_bkg"], self.plotdata["bkg_data"], xerr=self.plotdata["eng_bkg_err"], yerr=self.plotdata["bkg_data_err"], fmt='.', color="red", label="bkg data", alpha=0.35, lw=1.2)
            self.total_model_line, = self.ax1.plot(self.plotdata["eng_src"], self.total_model, label="Full model", lw=1.8)
            self.total_bkg_line, = self.ax1.plot(self.plotdata["eng_bkg"], self.total_bkg_model, label="Bkg model", lw=1.6, color="grey")
            if Plot.nAddComps(2) < Plot.nAddComps(1):
                self.total_src_line, = self.ax1.plot(self.plotdata["eng_src"], self.total_src_model, label="Src model", lw=1.6, color="purple")
            energy_type = 'eng_src'
        elif self.xspec_type == 'src':
            self.ax1.errorbar(self.plotdata["eng_src"], self.plotdata["src_data"], xerr=self.plotdata["eng_src_err"], yerr=self.plotdata["src_data_err"], fmt='.', color="black", label="src data", alpha=0.5, lw=1.2)
            self.total_model_line, = self.ax1.plot(self.plotdata["eng_src"], self.total_model, label="Full model", lw=1.8)
            energy_type = 'eng_src'
        elif self.xspec_type == 'bkg':
            self.ax1.errorbar(self.plotdata["eng_bkg"], self.plotdata["bkg_data"], xerr=self.plotdata["eng_bkg_err"], yerr=self.plotdata["bkg_data_err"], fmt='.', color="black", label="bkg data", alpha=0.5, lw=1.2)
            self.total_model_line, = self.ax1.plot(self.plotdata["eng_bkg"], self.total_model, label="Bkg model", lw=1.8)
            energy_type = 'eng_bkg'
            self.PiB_line, = self.ax1.plot(self.plotdata["eng_bkg"], self.Pib_model, label="PIB model", linestyle=":")
            
        if self.xspec_type == 'src_bkg' and Plot.nAddComps(2) == Plot.nAddComps(1):
            self.component_lines, = self.ax1.plot(self.plotdata[energy_type], self.total_src_model, label=self.label_list[0], linestyle="--")
        elif self.xspec_type == 'src' and Plot.nAddComps() == 0:
            self.component_lines, = self.ax1.plot(self.plotdata[energy_type], self.total_model, label=self.label_list[0], linestyle="--", lw=1)
        else:
            for i in range(len(self.label_list)):
                line, = self.ax1.plot(self.plotdata[energy_type], Plot.addComp(i+1, 1), label=self.label_list[i], linestyle="--")
                self.component_lines.append(line)
            
        self.ax1.set_yscale("log")
        self.ax1.set_xscale("log")
        self.ax1.set_ylabel(r"$\mathrm{Counts\ s^{-1}\ keV^{-1}}$")
        y_min, y_max = np.min(Plot.y(1)), np.max(Plot.y(1))
        self.ax1.set_ylim(y_min/5, y_max*5)
        self.ax1.legend()
        self.res_points = self.ax2.errorbar(self.plotdata[energy_type], self.residuals, xerr=self.plotdata[f"{energy_type}_err"], yerr=self.residuals_err, fmt='.', color="black", label="residuals", alpha=0.5, lw=1.2)
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
        self.set_values(new_values, frozen_mask=False)
        self.C = Fit.statistic
        self.dof = Fit.dof
        Plot('ldata delchi')
        
        if self.xspec_type == 'src_bkg':
            self.total_model = Plot.model(1)
            self.total_src_model = np.zeros(len(Plot.model(1)))
            if Plot.nAddComps(2) < Plot.nAddComps(1):
                for i in range(1, Plot.nAddComps(1)-Plot.nAddComps(2) + 1):
                    self.total_src_model += np.array(Plot.addComp(i, 1))
                self.total_bkg_model =Plot.model(2)
                self.total_src_line.set_ydata(self.total_src_model)

            if Plot.nAddComps(2) == Plot.nAddComps(1):
                self.total_bkg_model = np.zeros(len(Plot.model(1)))
                for i in range(1, Plot.nAddComps()):
                    self.total_bkg_model += np.array(Plot.addComp(i, 1))
                self.total_src_model = self.total_model - self.total_bkg_model
                self.total_bkg_model = Plot.model(2)   
                self.component_lines.set_ydata(self.total_src_model)
            self.total_bkg_line.set_ydata(self.total_bkg_model)
        
        elif self.xspec_type == 'src':
            self.total_model = Plot.model(1)
            
        elif self.xspec_type == 'bkg':
            self.total_model = Plot.model(1)
            self.Pib_model = np.zeros(len(Plot.model(1)))
            for i in range(len(self.label_list)+1, Plot.nAddComps() + 1):
                self.Pib_model += np.array(Plot.addComp(i, 1))
            self.PiB_line.set_ydata(self.Pib_model)

        self.residuals = Plot.y(1, 2)
        self.residuals_err = Plot.yErr(1, 2)
    
        self.total_model_line.set_ydata(self.total_model)
        if not(self.xspec_type == 'src_bkg' and Plot.nAddComps(2) == Plot.nAddComps(1)):
            if self.xspec_type == 'src' and Plot.nAddComps() == 0:
                self.component_lines.set_ydata(self.total_model)
            else:
                for i in range(len(self.label_list)):
                    self.component_lines[i].set_ydata(Plot.addComp(i + 1, 1))
            
        self.ax1.set_title(f"CSTAT/d.o.f. = {(self.C / self.dof):.2f} ({self.C:.2f}/{self.dof})")
        update_errorbar(self.res_points, Plot.x(1, 2), self.residuals, xerr=Plot.xErr(1, 2), yerr=self.residuals_err)
        self.fig.canvas.draw_idle()

    def create_sliders(self):
        self.sliders = [
            FloatSlider(
                min=max(bounds[0], val - abs(val*3)),
                max=min(bounds[1], val + abs(val*3)),
                # min=bounds[0],
                # max=bounds[1],
                value=val,
                step=val / 10,
                continuous_update=True,
                description=f'{model}.{comp}.{param}:',  # Add a colon for better spacing
                style={'description_width': '150px'},  # Adjust description width for more space
                readout_format='.2e' if abs(val) < 1e-2 or abs(val) > 100 else '.2f',  # Use scientific notation for small/large numbers
                layout=widgets.Layout(width='400px')  # Make sliders bigger
            )
            for model, comp, param, val, bounds in zip(self.model_type, self.comp_names, self.param_names, self.values, self.bounds)
        ]

    def create_interactive_layout(self):
        self.create_sliders()
        def proxy_function(**kwargs):
            ordered_values = [kwargs[k] for k in sorted(kwargs.keys())]
            return self.update_model(ordered_values)

        controls = {f'v{i:02}': slider for i, slider in enumerate(self.sliders[:99])}
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
        self.set_model()
        self.extract_parameters()
        self.final_set()
        self.create_plot()
        self.create_interactive_layout()
    
    def run_with_updated_model(self):
        self.extract_parameters()
        self.final_set()
        self.create_plot()
        self.create_interactive_layout()