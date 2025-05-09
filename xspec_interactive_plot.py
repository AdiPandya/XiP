import matplotlib.pyplot as plt
import ipywidgets as widgets
from xspec import *
import numpy as np
import os
from xspec import AllData, AllModels, Spectrum, Xset, Model
from ipywidgets import FloatSlider, VBox, HBox, interactive_output
from IPython.display import display

def get_free_parameters(AllModels, AllData):
    """
    Extract names and values of free (unfrozen, unlinked) parameters
    for all defined models in a multi-spectrum XSPEC analysis.

    Parameters:
        AllModels: XSPEC model object (e.g., AllModels)
        AllData: XSPEC data object (e.g., AllData)

    Returns:
        free_names: list of arrays of parameter names (free only)
        free_values: list of arrays of parameter values (free only)
    """
    free_names = []
    free_values = []

    Models = [AllModels.sources[k] for k in sorted(AllModels.sources.keys())]
    N_spec = AllData.nSpectra

    for n in range(1, N_spec + 1):
        for mod in Models:
            try:
                m = AllModels(n, mod)
            except:
                free_names.append(np.array([]))
                free_values.append(np.array([]))
                continue

            froz = np.array([m(i + 1).frozen for i in range(m.nParameters)])
            links = np.array([m(i + 1).link != '' for i in range(m.nParameters)])
            is_frozen_or_linked = froz | links

            pars = np.array([m(i + 1).values[0] for i in range(m.nParameters)])
            names = np.array([f"m_{mod}.{comp}.{param}" 
                              for comp in m.componentNames 
                              for param in getattr(m, comp).parameterNames])
            # Filter to only free parameters
            free_names.append(names[~is_frozen_or_linked])
            free_values.append(pars[~is_frozen_or_linked])

    # Flatten the lists
    free_names = np.concatenate(free_names)
    free_values = np.concatenate(free_values)

    return free_names, free_values

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
    


def interactive_xspec_plot(Dir, bkgfile, rmf, arf, model_name, param_defaults, label_list, energy_range=(0.2, 8.0)):
    """
    Create an interactive plot for an XSPEC model.

    Parameters:
        model_name (str): The XSPEC model name (e.g., 'apec+tbabs(apec+powerlaw)').
        param_defaults (dict): list of default values.
        energy_range (tuple): The energy range for the plot (default: (0.2, 8.0)).
        spectrum_data (tuple): A tuple containing energy and flux arrays for the spectrum data (default: None).
    """
    Xset.chatter = 0
    Xset.xsect = 'vern'
    Xset.abund = 'wilm'
    Fit.statMethod = 'cstat'
    
    Xset.allowPrompting = False
    # Initialize XSPEC model
    AllModels.clear()
    AllData.clear()
    pwd = os.getcwd()
    os.chdir(Dir)

    sp_bkg = Spectrum(bkgfile)
    sp_bkg.multiresponse[0] = rmf
    sp_bkg.multiresponse[0].arf = arf
    sp_bkg.multiresponse[1] = rmf
    os.chdir(pwd)
    
    AllData.ignore(f"**-{energy_range[0]},{energy_range[1]}-**")
    
    # Xset.chatter = 10
    # AllData.show()
    
    m_bkg = Model(model_name, "bkg", 1)
    # Load the model m_fwc
    Xset.restore(f"{Dir}TM8_FWC_c010_mod_customized_bkg.dat")
    m_fwc = AllModels(1, "fwc")

    # Freeze all parameters of m_fwc except the constant
    for _name in m_fwc.componentNames:
        if _name != "constant":
            _comp = m_fwc.__getattribute__(_name)
            for _pname in _comp.parameterNames:
                _par = _comp.__getattribute__(_pname)
                _par.frozen = True
    
    # global comp_names, param_names, values
    names, _ = get_free_parameters(AllModels, AllData)
    values = param_defaults
    if len(names) != len(values):
        raise ValueError(f"Number of parameters ({len(names)}) does not match number of values ({len(values)})")
    model_type = [name.split('.')[0] for name in names]
    comp_names = [name.split('.')[1] for name in names]
    param_names = [name.split('.')[2] for name in names]
    
    def set_values(model_type, comp_names, param_names, set_values):
        """
        Set the values of the parameters in the XSPEC model.
        """
        for i in range(len(set_values)):
            if model_type[i] == "m_bkg":
                model_obj = m_bkg
            elif model_type[i] == "m_fwc":
                model_obj = m_fwc
            try:
                component = getattr(model_obj, comp_names[i])
                setattr(component, param_names[i], set_values[i])
            except Exception as e:
                print(f"Error setting parameter {model_type[i]}.{comp_names[i]}.{param_names[i]}: {e}")

    set_values(model_type, comp_names, param_names, values)
    
    Fit.query = "yes"
    Fit.nIterations = 1000
    Fit.perform()
    
    _, values = get_free_parameters(AllModels, AllData)
    # print(values_fit)
    set_values(model_type, comp_names, param_names, values)
    C = Fit.statistic
    dof = Fit.dof
    Plot.device = "/null"
    Plot.add = True
    Plot.xAxis = "keV"
    Plot.setRebin(minSig=20, maxBins=100)
    Plot('ldata delchi')
    bkg_data = Plot.y(1)
    bkg_data_err = Plot.yErr(1)
    eng_bkg= Plot.x(1)
    eng_bkg_err = Plot.xErr(1)
    
    total_model = Plot.model(1)
        
    res_bkg = Plot.y(1,2)
    res_bkg_err = Plot.yErr(1,2)
    Pib_model = np.zeros(len(Plot.model(1)))   
    for i in range(Plot.nAddComps()-26, Plot.nAddComps()-1):
        Pib_model += np.array(Plot.addComp(i, 1))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0})
    
    ax1.errorbar(eng_bkg, bkg_data, xerr=eng_bkg_err, yerr=bkg_data_err, fmt='.', color="black", label="bkg data", alpha=0.5, lw=1.2)
    
    total_model_line, = ax1.plot(eng_bkg, total_model, label="Full model")
    
    component_lines = []
    
    for i in range(1, Plot.nAddComps() - 26):
        line, = ax1.plot(eng_bkg, Plot.addComp(i, 1), label=label_list[i - 1], linestyle="--")
        component_lines.append(line)
        
    PiB_line, =ax1.plot(eng_bkg, Pib_model, label="PIB model", linestyle="--")
        
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.set_ylabel(r"$\mathrm{Counts\ s^{-1}\ keV^{-1}}$")
    ax1.set_ylim(3, 2e2)
    ax1.legend()
    
    # Plot the residuals
    res_points = ax2.errorbar(eng_bkg, res_bkg, xerr=eng_bkg_err, yerr=res_bkg_err, fmt='.', color="black", label="residuals", alpha=0.5, lw=1.2)
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.set_xscale("log")
    ax2.set_xlabel("Energy [keV]")
    ax2.set_ylabel("Residuals")
    ax2.set_xticks([0.2, 0.5, 1, 2, 5])
    ax2.set_xticklabels(['0.2', '0.5', '1', '2', '5'])
    ax2.set_ylim(-5, 5)
    ax2.set_yticks([-3, 0, 3])
    ax2.axhspan(-3, 3, color='gray', alpha=0.15)
    ax1.set_title(f"CSTAT/d.o.f. = {(C/dof):.2f} ({C:.2f}/{dof})")


    def update_model(new_values):
        set_values(model_type, comp_names, param_names, new_values)

        C = Fit.statistic
        dof = Fit.dof

        Plot('ldata delchi')
        total_model = Plot.model(1)
        
        res_bkg = Plot.y(1,2)
        res_bkg_err = Plot.yErr(1,2)
        Pib_model = np.zeros(len(Plot.model(1)))   
        for i in range(Plot.nAddComps()-26, Plot.nAddComps()-1):
            Pib_model += np.array(Plot.addComp(i, 1))

        total_model_line.set_ydata(total_model)
        
        if Plot.nAddComps()-27 == len(label_list):
            pass
        else:
            print(f"Warning: The number of components in the model ({Plot.nAddComps()-27}) does not match the number of labels provided ({len(label_list)})")
        
        # Plot the data and models
        for i in range(1, Plot.nAddComps()-26):
            component_lines[i-1].set_ydata(Plot.addComp(i, 1))
        PiB_line.set_ydata(Pib_model)
        ax1.set_title(f"CSTAT/d.o.f. = {(C/dof):.2f} ({C:.2f}/{dof})")

        update_errorbar(res_points, eng_bkg, res_bkg, xerr=eng_bkg_err, yerr=res_bkg_err)
        
        fig.canvas.draw_idle()

    # Create the sliders
    sliders = [
        FloatSlider(
            min=val * 0.5,
            max=val * 2,
            value=val,
            step=val / 10,
            continuous_update=True,
            description=f'{comp}.{param}',
            readout_format='.2f'
        )
        for comp, param, val in zip(comp_names, param_names, values)
    ]

    # Proxy that extracts values from kwargs in order of slider keys
    def proxy_function(**kwargs):
        # Sort keys to maintain slider order (v0, v1, ...)
        ordered_values = [kwargs[k] for k in sorted(kwargs.keys())]
        return update_model(ordered_values)

    # Control dictionary: {'v0': slider0, 'v1': slider1, ...}
    controls = {f'v{i}': slider for i, slider in enumerate(sliders)}

    # Bind to interactive output
    output = interactive_output(proxy_function, controls)

    # Add a reset button
    reset_button = widgets.Button(description="Reset", button_style="info")

    def reset_parameters(_):
        for slider, default_value in zip(sliders, values):
            slider.value = default_value
        update_model(values)

    reset_button.on_click(reset_parameters)

    layout = VBox([HBox([VBox(sliders), reset_button]), output])
    display(layout)