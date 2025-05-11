import numpy as np

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
