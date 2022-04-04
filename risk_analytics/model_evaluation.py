#!/usr/bin/env python
# -*- coding: UTF-8 -*-



# Import Python Modules
import numpy as np
from scipy.stats import multivariate_t, multivariate_normal


# Custom Modules
from risk import pandas_patched as pd
from risk import utils as ut
from risk import econometrics as ec
from trade_republic.risk_models import mc_vaR as mcv
from tools.charting import charting as CH




def os_convergence_test(rets, wgts, qtl, cut_ddate, model_fmt = {'model':'mc','distr':'t'}, n_sim = 5e2, repetitions = 25,
                   sim_len = [250, 500, 750, 1000], verbose = False):
    """
    Convergence Test of MonteCarlo Distribution regarding overshoots (daily loss > VaR quantile
    @param rets:
    @param wgts:
    @param qtl:
    @param n_sim:
    @param repetitions:
    @param sim_len:
    @param distr:
    @param verbose:
    @return:
    """
    # ToDo: Review Logic of the Test
    # Adjust Return series sby cut_ddate:
    rets_cut = rets.copy()[:cut_ddate]
    realized_loss = rets_cut.dot(wgts.T).iloc[-1][0]
    # Shift by 1 Day as the VaR needs be forecasted on cut_ddate:
    rets_cut = rets_cut[:-1]

    # Start Forecasting VaR by Multiple Repetitions
    out_ll = []
    for n in sim_len:
        if verbose: print("Running MonteCarlo Simulation with sample size sim_len = %s"%n)
        tmp_ll = []
        for rep in range(0, repetitions):
            if verbose: print("\t Repetition num = %s"%rep)
            if model_fmt['model'] == 'mc':
                sim_rets, sim_ts = mcv.mc_simulate(rets_cut[-n:], n = n_sim, distr = model_fmt['distr'])
            tmp_vaR = ec.sim2VaR(sim_rets.dot(wgts.T), qtl, 1, calc_rets=False)
            os_bool = 1 if tmp_vaR.median().iloc[0] > realized_loss else 0
            tmp_ll.append(os_bool)
        # Calculate the probability of underestimating the value at risk quantile (= overshoot)
        prob_crit = len([k for k in tmp_ll if k == 1])/len(tmp_ll)
        out_ll.append(prob_crit)

    # Format Data
    medians = [np.median(k) for k in out_ll]
    means = [np.mean(k) for k in out_ll]
    y_axis = np.asarray(out_ll)
    x_axis = np.asarray(sim_len)

    for x, y in zip(x_axis, y_axis):
        CH.scatter([x] * len(y), y, s=12)

    for x, y in zip(n_sim, means):
        CH.scatter(x, y, s=12, color = 'red')

    for x, y in zip(n_sim, medians):
        CH.scatter(x, y, s=12, color = 'blue')

    for x, y in zip(n_sim, len(n_sim)*[rets.dot(wgts.T).iloc[:, 0].quantile(0.01)]):
        CH.scatter(x, y, s=12, color = 'yellow')

    #CH.xscale('log')
    CH.title("Direct Monte-Carlo Estimation")
    CH.ylabel("Probability Estimate")
    CH.xlabel('Number of Samples')
    CH.grid(True)
    CH.show()



def model_backtest(rets, wgts, qtl, start = -251, model_fmt = {'model':'mc','distr':'norm', 'sim_len':250}, n_sim = 100, verbose = True):
    """
    Rolling window Model backtest run for testing and reproduction purposes. The output aligns realized returns and forecasted 1D VaR
    figures. The data can be plotted via
    @param rets: dataframe
    @param wgts: numpy array, len(wgts) == len(rets)
    @param qtl: e.g. 0.99 for VaR99
    @param start: int, begin of the rolling window. default = -251
    @param model_fmt: MonteCarlo - {'model':'mc','distr':'t', 'sim_len':250}
    @param n_sim: number of simulated scenarios
    @param verbose: bool, used for printing
    @return: dataframe with aligned realized returns and VaR forecasts
    """
    #ToDo: Implement Clean and Dirty VaR App

    # Output DataFrame
    out_var_1d = pd.DataFrame(index = rets.index[start:-1], columns = ['var%s_1D'%str(int(100*qtl))])

    # Start Forecasting VaR on rolling window
    for n in range(len(out_var_1d.index)):
        if verbose: print("Forecasting value at risk for %s"%out_var_1d.index[n])
        # Slice the returns series:
        sim_len = model_fmt['sim_len']
        tmp_rets = rets[:out_var_1d.index[n]][-sim_len:]

        # Selecting the model and run simulation
        if model_fmt['model'] == 'mc':
            sim_rets, sim_ts = mcv.mc_simulate(tmp_rets, n = n_sim, distr = model_fmt['distr'])
            tmp_vaR = ec.sim2VaR(sim_rets.dot(wgts.T), qtl, 1, calc_rets = False)
            out_var_1d.loc[out_var_1d.index[n]] = tmp_vaR.median().iloc[0]

    # Create Output with return series:
    out_var_1d.loc[rets.index[-1], 'var99_1D'] = np.nan # Used for shifting as the latest VaR figure is a forecast
    out = rets.dot(wgts.T)[start:].join(out_var_1d.shift(1))

    # Format Output:
    out.columns = ['returns','var99_1d']
    out = out.dropna() # Drops first roe, which is NaN due to shifting the vaR forecast

    return out
