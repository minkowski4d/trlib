#!/usr/bin/env python
# -*- coding: UTF-8 -*-


# Import Python Modules
import numpy as np

# Custom Modules
from trlib import pandas_patched as pd
from trlib import econometrics as ec



def portfolio_vaR(rets_orig, wgts, holding_period=1, engine='mc', fmt_engine={}, verbose=True):
    """
    Random Weights
    wgts = np.random.dirichlet(np.ones(len(rets.columns)),size=1)
    """

    # Import multiple used models:
    # Python
    from scipy.stats import norm, t

    # Custom Modules
    from trlib.risk_models import support_models as smo


    # ToDo: Implement quick Data shape print

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

        fmt_engine_keys = ['window', 'qtl', 'holding_period', 'rescale', 'decay', 'ftol', 'max_iter', 'fhs', 'garch_engine']

        # Default Values
        fmt_engine_default = {'window_default': 250,
                              'qtl_default': 0.01,
                              'holding_period_default': 1,
                              'rescale_default': 1000,
                              'decay_default': 1,
                              'ftol_default': 1e-06,
                              'max_iter_default': 150,
                              'fhs_default': False,
                              'garch_engine_default': engine}


    elif engine == 'mc':
        # Monte Carlo Model Inputs
        # window: lookback period, e.g. 250 days
        # n_sim: number of simulations

        fmt_engine_keys = ['window', 'holding_period', 'n_sim', 'distr', 'decay', 'ewma_cov', 'qtl']
        # Default Values
        fmt_engine_default = {'window_default': 250,
                              'holding_period_default': 1,
                              'qtl_default': 0.01,
                              'n_sim_default': 1000,
                              'distr_default': 'norm',
                              'decay_default': 0.94,
                              'ewma_cov_default': True,
                              'qtl': 0.01}

    elif engine == 'hs':
        fmt_engine_keys = ['window', 'qtl', 'holding_period', 'decay', 'distr']

        # Default Values
        fmt_engine_default = {'window_default': 250,
                              'qtl_default': 0.01,
                              'holding_period_default': 1,
                              'decay_default': 0.94,
                              'distr': 'norm'}

    # Build fmt_engine:
    for k in fmt_engine_keys:
        if not k in fmt_engine.keys():
            fmt_engine[k] = fmt_engine_default[k + '_default']

    fmt_engine_string = ''
    for k in fmt_engine:
        fmt_engine_string += '\t\t\t%s: %s\n'%(k, fmt_engine[k])

    # Initialising Risk Model Calculation
    # Slice returns:
    rets = rets_orig.copy()
    rets = rets[-fmt_engine['window']:]

    if verbose:
        if verbose:
            # Print Model Setup
            print("\n     ------------------------------------------------------------")
            print("               Portfolio VaR Calculation")
            print("     ------------------------------------------------------------")
            print('      Model: \n\t\t\t%s\n'%engine)
            print('      Model Inputs:\n %s'%fmt_engine_string)
            print('      Window Start and End:\n \t\t\tStart %s, End %s'%(rets.index[0].date(), rets.index[-1].date()))
            print("     ------------------------------------------------------------")

    # Calculate Value at Risk
    # GARCH
    if engine == 'egarch' or engine == 'gjr':
        # Build portfolio
        ret_vaR = rets.portfolio_rets(wgts)
        # Initialise GARCH Calculation
        out = gm.calculate_vaR_garch(ret_vaR[['pf']],
                                        qtl = fmt_engine['qtl'],
                                        holding_period = holding_period,
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
        # Import Model
        from trlib.risk_models import mc_models
        if verbose : print("\nInitializing MonteCarlo Simulation for Value at Risk Calculation")
        # Initialise MonteCarlo Calculation
        sim_rets, sim_ts = mc_models.mc_simulate(rets = rets,
                                                 n = fmt_engine['n_sim'],
                                                 sim_len = fmt_engine['window'],
                                                 distr = fmt_engine['distr'],
                                                 ewma_cov = True,
                                                 verbose = verbose)

        # Build Output
        out = pd.DataFrame(index=['VaR'])
        # Calculate Value at Risk for holding period and quantile
        qtl = fmt_engine['qtl']
        h = fmt_engine['holding_period']
        tmp_vaR = sim_rets.dot(wgts.T).groupby("sim_id").quantile(qtl) * np.sqrt(h)
        tmp_vaR.columns = ['var%s_%sd' % (str(int(100 * (1-qtl))), str(h))]
        out.loc['VaR', 'var%s_%sd' % (str(int(100 * (1-qtl))), str(h))] = tmp_vaR.median().iloc[0] * np.sqrt(h)

    # Historical Simulation
    elif engine == 'hs':
        if verbose: print("\nInitializing Historical Simulation for Value at Risk Calculation")
        # Build portfolio
        ret_vaR = rets.portfolio_rets(wgts)

        # Apply EWMA volatility
        vol_ewma = smo.ewma_volatility(ret_vaR[-fmt_engine['window']:], decay=fmt_engine['decay'])

        # Calculate VaR
        qtl = fmt_engine['qtl']
        h = fmt_engine['holding_period']
        if fmt_engine['distr'] == 'norm':
            mu_norm, sig_norm = norm.fit(ret_vaR)
            value_at_risk = -mu_norm * h + vol_ewma * norm.ppf(qtl) * np.sqrt(h)
        elif fmt_engine['distr'] == 't':
            mu_norm, sig_norm = norm.fit(ret_vaR)
            nu, mu_t, sig_t = t.fit(ret_vaR)
            value_at_risk = - mu_norm * h + (h*(nu - 2)/nu) ** 0.5*t.ppf(qtl, nu) * vol_ewma

        # Build Output
        out = pd.DataFrame(index=['VaR'])
        out.loc['VaR', 'var%s_%sd'%(str(int(100*(1 - qtl))), str(holding_period))] = value_at_risk.iloc[0][0]

    return out



