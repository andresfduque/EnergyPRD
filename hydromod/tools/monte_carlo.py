"""
Monte Carlo Simulation of hydrological models


Author:
Saul Arciniega Esparza
Hydrogeology Group, Faculty of Engineering,
National Autonomous University of Mexico
zaul.ae@gmail.com | sarciniegae@comunidad.unam.mx
"""

#%% Import libraries
import sys
import numpy as np
import pandas as pd
from . import metrics
from copy import deepcopy


# ==============================================================================
# Requirements
# ==============================================================================

METRICS = {
    "cc": metrics.correlation,
    "nse": metrics.nash_sutcliffe_efficiency,
    "lnse": metrics.log_nash_sutcliffe_efficiency,
    "kge": metrics.kling_gupta_efficiency,
    "r2": metrics.determination_coeff,
    "mae": lambda yobs, ysim: -metrics.mean_absolute_error(yobs, ysim),
    "rmse": lambda yobs, ysim: -metrics.root_mean_square_error(yobs, ysim),
    "bias": metrics.bias,
}


def _lognormal_scale(vmin, vmax):
    # random number generation using logarithmic scale
    return np.exp(np.random.uniform(np.log(vmin), np.log(vmax)))


def _create_output_timeseries(save_vars, keep_best, start, end, freq):
    # create ouput timeseries from MonteCarlos Simulation
    start_date = start
    end_date = end
    freq = freq.upper()[0]

    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    if freq == "M":
        dates -= pd.offsets.MonthBegin()
    elif freq == "A":
        dates -= pd.offsets.YearBegin()

    save_outputs = {}
    for key in save_vars:
        save_outputs[key] = pd.DataFrame(
            np.zeros((len(dates), keep_best), dtype=np.float32),
            columns=np.arange(1, keep_best + 1),
            index=dates
        )
    return save_outputs


def timeserie_resample(serie, freq, agg):
    # Time series aggregation for scores calculation
    if freq is None:
        return serie
    if freq.lower() == "1d":
        return serie
    else:
        if agg == "mean":
            new_serie = serie.resample(freq).mean()
        else:
            new_serie = serie.resample(freq).sum()
        if "y" in freq.lower():
            new_serie.index = new_serie.index.year
        elif "m" in freq.lower():
            new_serie.index = new_serie.index - pd.offsets.MonthBegin()
        return new_serie


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    # Call in a loop to create terminal progress bar
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


# ==============================================================================
# Monte Carlo Class
# ==============================================================================

def MonteCarlo(model, forcings, param_bounds, numsimul=1000, param_scale=None,
               save_vars=None, xobs=None, scores=None, keep_best=100, **kargs):
    """
    Monte Carlo simulations for uncertainty analysis in hydrological models

    :param model:          [class] input hydrological model
    :param forcings:       [DataFrame] input forcings with column names depending of the
                             requirements of the model
    :param param_bounds:   [dict] bounds for every parameter to generate random sample.
                             The dict keys must match the name of the model's parameters.
                             Example: {'ks': (min1, max1), 'alpha': (min2, max2)}
    :param numsimul:       [int] number of simulations. By default 1000
    :param param_scale:    [dict] scale for parameters generation. Options are 'linear' (default)
                             and 'log'. The dict keys must match the keys of the param_bounds.
                             Example: {'ks': 'log'}
    :param save_vars:      [list, tuple] list of simulation variables to save. If None (default),
                            any variable is saved
    :param xobs:           [DataFrame] time series with observations, where columns must match with
                             column names from model simulations.
    :param scores:         [list] options for error estimation from simulations. Options are set as
                            dictionaries with the following keys andvalues:
                                var:    [string] variable contained in xobs to compute error
                                metric: [string, funtion] metric to maximize. Defined metrics are
                                        'cc' (correlation coefficient), 'nse' (Nash-Sutcliffe Efficiency),
                                        'lnse' (Nash-Sutcliffe Efficiency using logarithms),
                                        'kge' (Kling-Gumpta Efficiency), 'r2' (determination coefficient),
                                        'mae' (mean absolute error), 'rmse' (root mean squared error),
                                        'bias' (1.0 - ((mean_sim / mean_obs - 1.0) ** 2) ** 0.5).
                                        User functions can be defined as fun(yobs, ysim) and must return a float value
                                weight: [float] metric weight to ponderate multiple options. If a unique
                                         metric is used, then weight must be 1.
                                step:   [string] (optional) time step for data aggregation. By default
                                         step=None (without aggregation). Additional units are 'd' (day),
                                         'W' (week), 'M' (month), 'Y' (year).
                                agg:    [string] (optional) time series aggregation method: 'mean' (dafault)
                                        or 'sum'. Parameters step and agg must be included
    :param keep_best:      [int] (optional) number of the best simulations to preserve according to
                             the general score.
    :param kwargs:         Additional inputs as 'start' and 'end'
    :return:               [dict] parameters, scores, simulations
    """

    sim_model = deepcopy(model)

    if xobs is None:
        keep_best = numsimul
    else:
        keep_best = min(keep_best, numsimul)

    save_outputs = {}
    if save_vars is not None:
        start_date = kargs.get("start", forcings.index.min())
        end_date   = kargs.get("end", forcings.index.max())
        freq = forcings.index.inferred_freq

        save_outputs = _create_output_timeseries(
            save_vars,
            keep_best,
            start_date,
            end_date,
            freq
        )

    model_params = sim_model.get_parameters()
    save_params = pd.DataFrame(
        np.zeros((keep_best, len(model_params)), dtype=np.float32),
        columns=model_params.keys(),
        index=np.arange(1, keep_best+1)
    )
    for col in model_params.keys():
        save_params.loc[:, col] = model_params[col]

    # Parameters ranges
    if param_scale is None:
        param_scale = {}

    parameters = dict.fromkeys(param_bounds.keys(), 0.0)
    parameters_generator = dict.fromkeys(param_bounds.keys())
    for key in param_bounds.keys():
        param_scale[key] = param_scale.get(key, "linear").lower()

    # Scores to maximize
    if scores is None or xobs is None:
        errors = pd.DataFrame([])
    else:
        # Check the consistency of metrics and observations
        tweigth, count, labels = 0, 0, []
        for j in range(len(scores)):
            if scores[j]["var"] in xobs.columns:
                count += 1
                if callable(scores[j]["metric"]):
                    pass
                elif type(scores[j]["metric"]) is str:
                    name = scores[j]["metric"].lower()
                    if name in METRICS:
                        scores[j]["metric"] = METRICS[name]
                    else:
                        raise ValueError("Unknown 'metric' name in scores!")
                else:
                    raise TypeError("'metric' in scores must be a string or function!")
                tweigth += scores[j]["weight"]

                if "scale" not in scores[j]:
                    #scores[j]["scale"] = "1d"
                    scores[j]["scale"] = None
                if "agg" not in scores[j]:
                    scores[j]["agg"] = "mean"

                label = "{}.{}".format(scores[j]["var"], count)
                labels.append(label)
                scores[j]["label"] = label
            else:
                raise ValueError("Variable {key} is not defined in scores or observations".format())
        # force tweigth=1
        residual = (1.0 - tweigth) / count
        for j in range(len(scores)):
            scores[j]["weight"] += residual

        # create output error variable
        error_cols = ["score"] + labels
        errors = pd.DataFrame(
            np.zeros((keep_best, len(error_cols)), dtype=np.float32),
            columns=error_cols,
            index=np.arange(1, keep_best+1)
        )
        errors.loc[:, "score"] = -9999.0
        error_vector = np.zeros(len(labels), dtype=np.float32)
        error_label = np.array(labels, dtype=object)

    # Main loop
    print_progress(0, numsimul, prefix='Progress:', suffix='Complete', bar_length=50)
    for i in range(numsimul):
        # random parameters
        for key in parameters.keys():
            if param_scale[key] == "log":
                parameters[key] = _lognormal_scale(
                    param_bounds[key][0],
                    param_bounds[key][1]
                )
            else:
                parameters[key] = np.random.uniform(
                    param_bounds[key][0],
                    param_bounds[key][1]
                )

        # run simulations using random parameters
        simulations = sim_model.run(
            forcings,
            start=kargs.get("start", None),
            end=kargs.get("end", None),
            **parameters
        )

        # compute errors
        idx = i + 1
        keep = True
        if scores is not None and xobs is not None:
            err = 0.0
            for j in range(len(scores)):
                key = scores[j]["var"]
                yobs = timeserie_resample(
                    xobs.loc[:, key],
                    scores[j]["scale"],
                    scores[j]["agg"]
                )
                ysim = timeserie_resample(
                    simulations.loc[:, key],
                    scores[j]["scale"],
                    scores[j]["agg"]
                )
                errvar = scores[j]["metric"](yobs, ysim)
                error_vector[j] = errvar
                error_label[j] = scores[j]["label"]
                err += scores[j]["weight"] * errvar
            # search minimum error
            idmin = errors.score.argmin()
            if err > errors.iloc[idmin, 0]:
                idx = idmin + 1
                errors.loc[idx, "score"] = err
                for j in range(len(error_label)):
                    errors.loc[idx, error_label[j]] = error_vector[j]
            else:
                keep = False

        # save outputs
        if save_outputs and keep:
            for key in save_outputs.keys():
                if key in simulations.columns:
                    save_outputs[key].loc[:, idx] = simulations.loc[:, key]
        if keep:
            for key in parameters.keys():
                save_params.loc[idx, key] = parameters[key]

        print_progress(i+1, numsimul, prefix='Progress:', suffix='Complete', bar_length=50)

    return {"parameters": save_params, "scores": errors, "simulations": save_outputs}


