#!/usr/bin/env python
# -*- coding: UTF-8 -*-



# Import Python Modules
import numpy as np


# Custom Modules
from trlib import pandas_patched as pd
from trlib import utils as ut
from scipy.stats import multivariate_t, multivariate_normal


def mc_simulate(rets, n=1000, sim_len=250, distr='norm', decay=1, verbose=True):
    """
    Simulation based on MonteCarlo Algorithm.
    @param rets: dataframe - underlying returns
    @param n: integer -  number of simulations, default = 1000
    @param sim_len: integer - length of simulation output, default = 250
    @param distr: string - multivariate distribution used in simulation, default = 'norm', options = 't' for Student T
    @param decay: default 1. Values 1 to 0.94
    @param verbose:

    Additional Info:
    - Plotting Results:
    For plotting the time series simulation results for a single underlying use
    -> sim_ts.iloc[:,[0]].unstack().T.reset_index().set_index('time_id').drop('level_0',axis=1).plot(legend = False)
    @return:
    """

    if verbose: print('\n Starting Simulation...')
    if verbose: pb = ut.ProgressBar(n)

    # Define Multivariate Inputs
    rets_mc = rets.copy()
    # Use decay
    if decay != 1:
        rets_mc = rets_mc.sort_index(ascending=False).ewm(alpha=decay, adjust=False).mean().sort_index(ascending=True)
    means = list(rets_mc.mean())

    rets_cov_mtx = np.matrix(rets_mc.cov())

    # Simulation Iteration
    sim_rets = pd.DataFrame(); sim_ts = pd.DataFrame()
    for i in  np.arange(0, n, 1):
        # Run Multivariate MonteCarlo based on 'distr' input.
        if distr == 'norm':
            tmp_rets = pd.DataFrame(multivariate_normal.rvs(means, rets_cov_mtx, size = sim_len), columns = rets.columns)

        elif distr == 't':
            tmp_rets = pd.DataFrame(multivariate_t.rvs(means, rets_cov_mtx, df=10, size = sim_len), columns = rets.columns)

        # Create Indexed Time Series:
        tmp_ts = tmp_rets.rets2lvl()

        # Reformat DataFrames for MultiIndex Output
        tmp_rets = tmp_rets.reset_index() ; tmp_ts = tmp_ts.reset_index()
        tmp_rets = tmp_rets.rename(columns = {'index':'time_id'}) ; tmp_ts = tmp_ts.rename(columns = {'index':'time_id'})
        # Introduce New Index for later MultiIndex Dataframe throughout "groupby"
        tmp_rets['sim_id'] = i ; tmp_ts['sim_id'] = i

        # Final Outputs
        sim_rets = pd.concat([sim_rets, tmp_rets.groupby(['sim_id', 'time_id']).sum()])
        sim_ts = pd.concat([sim_ts, tmp_ts.groupby(['sim_id', 'time_id']).sum()])

        # Progress Bar update
        if verbose: pb.animate(i)

    if verbose: print('\n ...Simulation Terminated\n')

    return sim_rets, sim_ts





