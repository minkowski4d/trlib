#!/usr/bin/env python
# -*- coding: UTF-8 -*-



# Import Python Modules
import numpy as np


# Custom Modules
from risk import pandas_patched as pd
from risk import econometrics as ec
from trade_republic.risk_models import mc_vaR as mcv



def portfolio_vaR(rets, wgts, qtl, horizons, vaR_model = 'mc', n = 1000, sim_len = 250, verbose = True):

    out = pd.DataFrame(index = ['VaR'])
    if vaR_model == 'mc':
        if verbose: print("\nInitializing MonteCarlo Simulation for Value at Risk Calculation")
        sim_rets, sim_ts = mcv.mc_simulate(rets = rets, n = n, sim_len = sim_len, verbose = verbose)
        for h in horizons:
            tmp_vaR = ec.sim2VaR(sim_rets.dot(wgts.T), qtl, h, calc_rets = False)
            tmp_vaR.columns = ['VaR %s,%sD'%(str(int(100*qtl)), str(h))]
            out.loc['VaR','VaR %s,%sD'%(str(int(100*qtl)), str(h))] = tmp_vaR.median().iloc[0]

    return out