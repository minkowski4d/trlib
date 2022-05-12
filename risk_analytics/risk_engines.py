#!/usr/bin/env python
# -*- coding: UTF-8 -*-


# Import Python Modules
import numpy as np

# Custom Modules
from trlib import pandas_patched as pd
from trlib import econometrics as ec



def portfolio_vaR(rets, wgts, holding_period=1, engine='mc', fmt_engine={}, verbose=True):
    """
    wgts = np.random.dirichlet(np.ones(len(rets.columns)),size=1)
    """

    # ToDo: Implement quick Data shape print
    rets_vaR = rets.copy()

    if engine == 'egarch' or engine == 'gjr':
        # Import Model
        from trlib.risk_models import garch_models as gm

        # GARCH Model Inputs
        # window: lookback period, e.g. 250 days
        # horizon: forecast horizon, e.g. 1
        # rescale: rescale for fitting purposes
        # decay: values from 1 to 0.94 for EWMA
        # ftol: tolerance for optimizer convergence, default 1e-06
        # max_iter: maximum iteration for convergence search

        fmt_engine_keys = ['window', 'qtl', 'horizon', 'rescale', 'decay', 'ftol', 'max_iter', 'fhs', 'garch_engine']

        # Default Values
        fmt_engine_default = {'window_default': 250,
                              'qtl_default': 0.01,
                              'horizon_default': 1,
                              'rescale_default': 1000,
                              'decay_default': 1,
                              'ftol_default': 1e-06,
                              'max_iter_default': 150,
                              'fhs_default': False,
                              'garch_engine_default': engine}


    elif engine == 'mc':
        # Import Model
        from trlib.risk_models import mc_models

        # Monte Carlo Model Inputs
        # window: lookback period, e.g. 250 days
        # n_sim: number of simulations

        fmt_engine_keys = ['window', 'n_sim', 'distr']
        # Default Values
        fmt_engine_default = {'window_default': 250,
                              'qtl_default': 0.01,
                              'n_sim_default': 1000,
                              'distr_default': 'norm'}

    # Build fmt_engine:
    for k in fmt_engine_keys:
        if not k in fmt_engine.keys():
            fmt_engine[k] = fmt_engine_default[k + '_default']

    fmt_engine_string = ''
    for k in fmt_engine:
        fmt_engine_string += '\t\t\t%s: %s\n'%(k, fmt_engine[k])

    if verbose:
        if verbose:
            # Print Model Setup
            print("\n     ------------------------------------------------------------")
            print("               Portfolio VaR Calculation")
            print("     ------------------------------------------------------------")
            print('      Model: \n\t\t\t%s\n'%engine)
            print('      Model Inputs:\n %s'%fmt_engine_string)
            print("     ------------------------------------------------------------")

    # Calculate Value at Risk
    # GARCH
    if engine == 'egarch' or engine == 'gjr':
        ret_vaR = rets.portfolio_rets(wgts)
        out = gm.calculate_vaR_garch(ret_vaR[['pf']],
                                        window = fmt_engine['window'],
                                        qtl = fmt_engine['qtl'],
                                        horizon = fmt_engine['horizon'],
                                        rescale = fmt_engine['rescale'],
                                        decay = fmt_engine['decay'],
                                        ftol = fmt_engine['ftol'],
                                        max_iter = fmt_engine['max_iter'],
                                        fhs = fmt_engine['fhs'],
                                        garch_engine = fmt_engine['garch_engine'])
        # Format
        out.index.name = 'ddate'

    # Monte Carlo
    elif engine == 'mc':
        if verbose : print("\nInitializing MonteCarlo Simulation for Value at Risk Calculation")
        sim_rets, sim_ts = mc_models.mc_simulate(rets = rets,
                                                 n = fmt_engine['n_sim'],
                                                 sim_len = fmt_engine['window'],
                                                 distr = fmt_engine['distr'],
                                                 verbose = verbose)

        # Build Output
        out = pd.DataFrame(index=['VaR'])
        # Calculate Value at Risk for holding period and quantile
        qtl = fmt_engine['qtl']
        tmp_vaR = sim_rets.dot(wgts.T).groupby("sim_id").quantile(qtl) * np.sqrt(holding_period)
        tmp_vaR.columns = ['var%s_%sd' % (str(int(100 * (1-qtl))), str(holding_period))]
        out.loc['VaR', 'var%s_%sd' % (str(int(100 * (1-qtl))), str(holding_period))] = tmp_vaR.median().iloc[0]

    return out



