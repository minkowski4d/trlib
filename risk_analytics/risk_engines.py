#!/usr/bin/env python
# -*- coding: UTF-8 -*-


# Import Python Modules
import numpy as np

# Custom Modules
from trlib import pandas_patched as pd
from trlib import econometrics as ec
from trlib.risk_models import mc_models


def portfolio_vaR(rets, wgts, qtl, horizons, engine='mc', fmt_engine={}, verbose=True):
    """
    wgts=np.random.dirichlet(np.ones(len(rets.columns)),size=1)
    """

    # ToDo: Implement quick Data shape print
    rets_vaR = rets.copy()

    if engine == 'egarch' or engine == 'gjr':

        # GARCH Model Inputs
        # window: lookback period, e.g. 250 days
        # horizon: forecast horizon, e.g. 1
        # rescale: rescale for fitting purposes
        # decay: values from 1 to 0.94 for EWMA
        # ftol: tolerance for optimizer convergence, default 1e-06
        # max_iter: maximum iteration for convergence search

        fmt_engine_keys = ['window', 'horizon', 'rescale', 'decay', 'ftol', 'max_iter', 'garch_engine']

        # Default Values
        fmt_engine_default = {'window_default': 250,
                              'horizon_default': 1,
                              'rescale_default': 1000,
                              'decay_default': 1,
                              'ftol_default': 1e-06,
                              'max_iter_default': 150,
                              'garch_engine_default': engine,
                              'eval_model_exec': "gm.calculate_vaR_garch(rets_var[['pf']], "\
                                                 "fmt_engine['window'], "\
                                                 "fmt_engine['horizon'], "\
                                                 "fmt_engine['rescale'], "\
                                                 "fmt_engine['decay'], "\
                                                 "fmt_engine['ftol'], "\
                                                 "fmt_engine['max_iter'], "\
                                                 "fmt_engine['garch_engine'])"}


    elif engine == 'mc':

        # Monte Carlo Model Inputs
        # window: lookback period, e.g. 250 days
        # n_sim: number of simulations

        fmt_engine_keys = ['window', 'n_sim', 'eval_model_exec']
        # Default Values
        fmt_engine_default = {'window_default': 250,'n_sim_default': 1000,
                              'eval_model_exec_default': "mc_models.mc_simulate(rets=rets_vaR, "\
                                                         "n=fmt_engine['n_sim'], "\
                                                         "sim_len=fmt_engine['window'], "\
                                                         "verbose=verbose)"}

    # Build fmt_engine:
    for k in fmt_engine_keys:
        if not k in fmt_engine.keys():
            fmt_engine[k] = fmt_engine_default[k + '_default']

    if verbose:
        fmt_engine_string = ''
        for k in fmt_engine:
            fmt_engine_string += '\t\t\t%s: %s\n'%(k, fmt_engine[k])

        if verbose:
            # Print Model Setup
            print("     ------------------------------------------------------------")
            print("               Portfolio VaR Calculation")
            print("     ------------------------------------------------------------")
            print('      Model: %s'%engine)
            print('      Model Inputs:\n %s'%fmt_engine_string)


    if verbose : print("\nInitializing MonteCarlo Simulation for Value at Risk Calculation")
    sim_rets, sim_ts = eval(fmt_engine['eval_model_exec'])

    out = pd.DataFrame(index=['VaR'])
    for h in horizons:
        tmp_vaR = ec.sim2VaR(sim_rets.dot(wgts.T), qtl, h, calc_rets=False)
        tmp_vaR.columns = ['VaR %s,%sD' % (str(int(100 * qtl)), str(h))]
        out.loc['VaR', 'VaR %s,%sD' % (str(int(100 * qtl)), str(h))] = tmp_vaR.median().iloc[0]


    return out



