#!/usr/bin/env python
# -*- coding: UTF-8 -*-


# Python Modules
import numpy as np
from numpy.linalg import inv
from scipy import stats as si

from trlib import pandas_patched as pd
from trlib import utils as ut


def WB(Y, p = 0.015, ff = 1, hori = 252):
    """
    Program for the Block Bootstrap on the series Y
    input: Y Series of dimension (T*N)sim2
           p size of the block; values from 0, 1; block size is equal to 1/p
           ff serves to have a fixed minimum size in the sampled data, i.e.
           resample weeks or months. if ff is one, then it starts by days (default)
           hori is the size of the resampled data S.
    """
    T, N = Y.shape

    LL = []  #LL variable will contain the length of each block
    S = np.zeros((hori, N))  #S contains the sample
    k = 0
    while k < hori:
        I = np.int(np.ceil(T*np.random.rand(1)))
        L = np.int(np.random.geometric(p))
        if k + L > hori: L = hori - k
        LL.append(L)
        #if I want to fix a frequency
        L = L - np.mod(L, ff)
        if I + L <= T:
            S[k:k + L, :] = Y[I:I + L, :]
        else:
            hj = I + L - T
            S[k:k + L - hj, :] = Y[I:T, :]
            S[k + L - hj + 1:k + L, :] = Y[1:hj, :]
        k = k + L
    return S, LL


def sim2rets(sim, retfreq = 1):
    m = sim.index.min()[1]
    return sim.pct_change(periods = retfreq).query("time_id>=%i"%(retfreq + m)).loc(axis = 0)[:, ::retfreq]


def sim2VaR(sim, p, h, calc_rets=False):
    if calc_rets:
        rets = sim2rets(sim, h)
        scale_t = 1
    else:
        rets = sim
        scale_t = np.sqrt(h)

    return rets.groupby("sim_id").quantile(1-p) * scale_t


def sim2cagr(sim, yd = 1440):
    rets = 1. + sim2TR(sim)
    rets = rets.applymap(lambda x: x ** (float(yd)/float(len(sim.index.levels[1]))) - 1.)
    return rets.astype(float)


def sim2TR(sim):
    min_t = sim.index.min()[1]
    max_t = sim.index.max()[1]
    rets = sim.loc(axis = 0)[:, (min_t, max_t)].pct_change().loc(axis = 0)[:, max_t]
    rets.index = rets.index.droplevel(1)
    return rets


def sim2vol(sim, retfreq = 5, calc_rets = True):
    factor = 260./retfreq
    if calc_rets:
        vols = sim2rets(sim, retfreq).groupby("sim_id").std()*np.sqrt(factor)
    else:
        vols = sim.groupby("sim_id").std()*np.sqrt(factor)
    return vols


def sim2MDD(sim):
    mdd_out = pd.DataFrame()

    print('Calculation in progress...')
    pb = ut.ProgressBar(len(sim.index.get_level_values(0).unique()))
    for i in range(0, max(sim.index.get_level_values(0))):
        temp = sim.loc[i]
        mdd_out.loc[i, sim.columns[0]] = -1*max((1 - temp/temp.cummax()).values)[0]
        # mdd = 0
        # for j in range(1, temp.shape[0]):
        #     peak = temp.iloc[:j].max().iloc[0]
        #     ip = temp.iloc[:j].idxmax().iloc[0]
        #     through = temp.loc[ip:].min().iloc[0]
        #     if through / peak - 1 < mdd:
        #         mdd = through / peak - 1

        # mdd_out.loc[i,sim.columns[0]]=mdd
        pb.animate(i)

    return mdd_out


def sim2sharpe(sim):
    #if len(sim)>252: raise Exception("Adjust code. Sharpe2Sim works on 252 observations max")
    shr_out = pd.DataFrame()
    for i in range(0, max(sim.index.get_level_values(0))):
        temp = sim.loc[i].pct_change().dropna()
        shr = (temp.mean()*len(temp))/(temp.std()*np.sqrt(len(temp)))
        shr_out.loc[i, 'Sharpe Ratio'] = shr.iloc[0]

    return shr_out


def sim2ts(ts_sim, wgts, verbose = False):
    sim_rets = sim2rets(ts_sim)
    out = pd.DataFrame(index = [0] + sim_rets.index.get_level_values(1).unique().tolist(),
                       columns = sim_rets.index.get_level_values(1).unique())
    for idx in sim_rets.index.get_level_values(0).unique():
        if idx%500 == 0 and verbose: print(idx)
        out[idx] = np.append([100], (1 + np.dot(sim_rets.loc[idx], wgts)).cumprod()*100)

    return out


def lpm(rets, threshold, order):
    threshold_array = np.empty(len(rets))
    threshold_array.fill(threshold)
    diff = threshold_array - rets.values
    diff = diff.clip(min = 0)
    return np.sum(diff ** order)/len(rets)


def hpm(rets, threshold, order):
    threshold_array = np.empty(len(rets))
    threshold_array.fill(threshold)
    diff = rets.values - threshold_array
    diff = diff.clip(min = 0)

    return np.sum(diff ** order)/len(rets)


def rolling_drawdown(df):

    out_dict = dict()
    run_mdd = pd.DataFrame(index = df.index)
    for col in df.columns:
        run_mdd['%s_run_mdd'%col] = df[col]/df[col].cummax() - 1

    out_dict['run_mdd'] = run_mdd

    # Calculate Max DrawDown
    out_dict['mdd'] = run_mdd.min().iloc[0]

    # Calculate Drawdown Length
    mdd_max_day = run_mdd.index[np.argmin(run_mdd.values)]
    peak_prior_max_mdd = run_mdd[run_mdd.values == 0][:mdd_max_day].index[-1]
    mdd_end = run_mdd.loc[mdd_max_day:].ne(0).idxmin()[0]
    out_dict['ddl'] = (mdd_max_day - peak_prior_max_mdd).days
    # Calculate Time To Recover:
    out_dict['t2r'] = (mdd_end - mdd_max_day).days

    return out_dict


def running_sharpe(df):
    ad = df.index_freq()  # average day length of returns
    ann = 365/ad  # for annualizing returns
    sqann = (365/ad) ** 0.5
    df.index.name = 'index'
    df = df.pct_change().reset_index().dropna()
    out = pd.DataFrame(index = list(range(1, len(df))), columns = ['ddate', 'run_shr'])
    for idx, row in df.iterrows():
        temp = df.loc[df['index'] <= row['index']]
        out.loc[idx, 'ddate'] = temp.loc[idx, 'index']
        if len(temp) >= 252:
            out.loc[idx, 'run_shr'] = ((temp.mean()*ann)/(temp.std()*sqann))[0]
        else:
            out.loc[idx, 'run_shr'] = np.nan

    return out.set_index('ddate')


def lsq(y, x, add_constant = True, stats = False):
    """
    quick and simple least squares implementation without relying on external packages
    """
    if add_constant:
        x = np.c_[np.ones(x.shape[0]), x]
    inv_xx = inv(np.dot(x.T, x))
    xy = np.dot(x.T, y)
    b = np.dot(inv_xx, xy)
    out = np.empty((3, x.shape[1]))

    out[0, :] = b
    if stats:
        df_e = y.shape[0] - x.shape[1]
        e = y - np.dot(x, b)
        sse = np.dot(e, e)/df_e
        out[2, :] = np.diagonal(sse*inv_xx)
        se = np.sqrt(out[2, :])
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            out[1, :] = b/se
        return out
    else:
        return out[0, :]


def option_delta(S, K, T, sigma, direction, rf, verbose = False):
    """
    Greeks Ouput
    :param S: Spot Price
    :param K: Strike Price
    :param T: Maturity in Days
    :param sigma: Volatility Underlying
    :param direction: call or put
    :param rf: RiskFree rate.
    :param verbose: print msgs
    :return:
    """

    import numpy as np
    import scipy.stats as si

    d1 = (np.log(S/K) + (rf + 0.5*sigma ** 2)*T/365)/(sigma*np.sqrt(T/365))
    if verbose: print(d1, S, K, T, sigma, direction)
    if direction == 'call':
        delta = si.norm.cdf(d1, 0, 1)
    if direction == 'put':
        delta = -si.norm.cdf(-d1, 0, 1)

    return delta


def option_price(S, K, T, sigma, direction, rf, verbose = False):
    from math import exp, log, sqrt

    d1 = (np.log(S/K) + (rf + 0.5*sigma ** 2)*T/365)/(sigma*np.sqrt(T/365))
    d2 = (np.log(S/K) + (rf - 0.5*sigma ** 2)*T/365)/(sigma*np.sqrt(T/365))

    if verbose: print(S, K, T, sigma, rf, direction)
    if direction == 'call':
        price = (S*si.norm.cdf(d1, 0.0, 1.0) - K*np.exp(-rf*T/365)*si.norm.cdf(d2, 0.0, 1.0))
    elif direction == 'put':
        price = (K*np.exp(-rf*T/365)*si.norm.cdf(-d2, 0.0, 1.0) - S*si.norm.cdf(-d1, 0.0, 1.0))

    return price


def rolling_corr(rets, window, x):
    df = rets.copy().loc[:, [x] + [l for l in rets.columns if l != x]]
    roll_corr = pd.DataFrame(index = df.index[window:], columns = df.columns)
    for k in range(window, len(df)):
        roll_corr.loc[df.index[k]] = df[k - window:k].corr().iloc[0]

    return roll_corr



def rolling_sharpe(rets, window):
    df = rets.copy()
    roll_sharpe = pd.DataFrame(index = df.index[window:], columns = df.columns)
    for k in range(window, len(df)):
        roll_sharpe.loc[df.index[k]] = df[k - window:k].shr()

    return roll_sharpe


def rolling_ret(ts, window):
    df = ts.copy()
    roll_ret = pd.DataFrame(index = df.index[window:], columns = df.columns)
    for k in range(window, len(df)):
        roll_ret.loc[df.index[k]] = df[k - window:k].ret().iloc[0]

    return roll_ret


def sim2cov(rets_sim):
    covs = rets_sim.groupby("sim_id").cov()
    covs = covs*250

    return covs




# def sim2beta(rets_sim, qtl, verbose = True):
#     out_beta = pd.DataFrame(columns = ['beta'])
#     for i in rets_sim.index.get_level_values(0).unique():
#         if i%250 == 0 and verbose: print(i)
#         tmp_res = ri.quantile_reg_sim(rets_sim.loc[i], qtl = qtl)
#         out_beta.loc[i, 'beta'] = tmp_res.params.iloc[1]
#
#     return out_beta


def qq_plot_dataset(rets, startdate=None, enddate=None, plot=False):

    rets_qq = rets.copy()
    if startdate: rets_qq = rets_qq[startdate:]
    if enddate: rets_qq = rets_qq[:enddate]

    rets_qq = rets_qq.fillna(0)
    out_qq = pd.DataFrame(columns = ['norm_qtl']+[k for k in rets.columns])
    out_qq['norm_qtl'] = si.probplot(rets_qq.iloc[:, 0].values, dist = "norm", fit = False, plot = None)[0]
    out_qq = out_qq.set_index('norm_qtl')
    for col in rets.columns:
        tmp = si.probplot(rets_qq[col].values, dist = "norm", fit = False, plot = None)
        out_qq[col] = tmp[1]

    if plot:
        fig_qq=CH.qq_plot(out_qq)
        return out_qq, fig_qq
    else:
        return out_qq


