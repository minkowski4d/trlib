#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import print_function
from pandas import *
from pandas.tseries.frequencies import to_offset
import traceback

__version__ = pandas.__version__

from datetime import datetime as _datetime
from datetime import date, timedelta
import six as _six
import numpy as np
import pytz

# Custom Modules
from trlib import utils as ut


def rebase(self, ip=None, rv=100, cut=True, fill=True):
    """
    rebases dataframe values by columns to a given value at a common point in time

    ip: point at which to rebase; if None, this is set to the first row without nan values
        may also be a date, in which case rebases at that date
        can also be a string: ytd, to rebase at beginning of current year
                              mtd, to rebase at beginning of current year
    rv: value at which to rebase; if this is a string, then the value of the corresponding
        column is used as rebase value
    cut: if true, removes the part before the rebasing point
    fill: if false, keep nan values
    """
    df = self.copy()
    index = df.index
    # if isinstance(index,core.indexes.datetimes.DatetimeIndex): index=index.date
    if ip is None:
        if isinstance(df, Series):
            ip = np.isnan(df.astype(np.float).values).argmin()
        else:
            ip = np.any(np.isnan(df.astype(np.float).values), 1).argmin()
    elif isinstance(ip, (date, _datetime, Timestamp)):
        if isinstance(ip, (_datetime, Timestamp)) and isinstance(df.index[0], date):
            ip = ip.date()
        elif isinstance(ip, date) and isinstance(df.index[0], (Timestamp, datetime)):
            ip = _datetime(ip.year, ip.month, ip.day).date()
        ip = (index >= ip).argmax()
    elif isinstance(ip, str):
        if ip.lower() == 'ytd':
            ip = (_datetime(index[-1].year, 1, 1) - offsets.BDay()).date()
        elif ip.lower() == 'mtd':
            ip = (_datetime(index[-1].year, df.index[-1].month, 1) - offsets.BDay()).date()
        else:
            ip = (index[-1] + to_offset(ip)).date()
        ip = (index >= ip).argmax()
    nans = df.isnull()
    if cut:
        df = df.iloc[ip:].fillna(method='ffill').fillna(method='bfill')
        ip = 0
    if not fill:
        df = df.mask(nans.reindex(df.index))
    if isinstance(rv, str):
        rv = df[rv].iloc[ip]
    df = df / df.iloc[ip] * rv
    return df


def sb_simulate(self, len_sim=None, data_in_levels=True, out_freq=0, n_sim=10000, ff=1, p=1. / 60., fpath=None,
                fname=None, to_lvl=True, verbose=1, seed=None):
    """

    :param self: dataframe, containing the series to be simulated
    :param len_sim: int, time horizon (length) of each simulation
    :param n_sim: int, number of simulations to be performed
    :param seed: int (between 0 and 2^32 - 1). set the seed of the random number generator. default = None. The seed is reset at the end of simulation
                back to the default value.
    :param data_in_levels: bools, set it to True if series passed are in levels and must be different
    :param out_freq: int, if 0 series simulated are returned in levels else they are returned as returns with freq=out_freq
    :param to_lvl: produce an indexed 100 timeseries out of simulate returns
    :param ff: int, min block size length
    :param p: float, probability of success for geometric distribution of blocks' length
    :param fpath: if not None it is a string stating where the simulation panel will be saved as pickle file
    :param fname: None or int, if int detrend simulating series (returns/yield changes) picking the cycle component of an HP filter
    :param verbose: bool, if not False print the % of sim processed

    :return: DataFrame, index: simulation_id/time_id --> columns: series simulated
    """
    from trlib.econometrics import WB
    # Prepare data for the stationary bootstrap analysis
    df = self.copy()
    df = df[df.index >= df.dropna().index[0]]
    df = df.fillna(method='ffill')
    if type(p) is not float: p = float(p)

    # If the dataframe passed contains series that are in levels then considered pct_changes at the highest frequency posible
    if data_in_levels is True:
        df = df.pct_change(periods=1).dropna()
    else:
        pass

    # If the length of the simulation is not specified then set it to the original series length
    if len_sim is None: len_sim = df.shape[0]

    # Initialize the output
    if out_freq < 1:
        len_sim = len_sim - 1
        l_out_sim = len_sim + 1 if to_lvl else len_sim
    else:
        l_out_sim = len(range(out_freq, len_sim + 1, out_freq))

    # Get values to be boostrapped
    values = df.values
    ids_list = list()
    results = list()
    i = 0
    if verbose > 0: print('Simulation in progress...')
    if verbose > 0: pb = ut.ProgressBar(n_sim)

    # Set the seed, default is None.
    np.random.seed(seed)
    for s in range(0, n_sim, 1):
        # ids for simulation items
        ids = list(zip([s] * l_out_sim, range(0, l_out_sim, 1)))

        # Generate a SB simulation
        sim, LL = WB(values, p=p, ff=ff, hori=len_sim)

        # Trasform the series of returns to series in levels
        if to_lvl:
            sim = 100. * np.vstack((np.ones(sim.shape[1]), np.cumprod(1. + sim, axis=0)))

            # If out_freq is 0 then keep the series in level, else produce returns at out_freq frequency
            if out_freq == 0:
                pass
            else:
                sim = (sim[out_freq::out_freq, :] / sim[0:-out_freq:out_freq, :]) - 1.

        results.append(sim)
        ids_list += ids

        i += 1
        if verbose > 0: pb.animate(i)

    # Resets the seed back to None.
    np.random.seed(None)
    sim_df = DataFrame(data=np.vstack(tuple(results)), index=MultiIndex.from_tuples(ids_list), columns=df.columns)
    # Ensure output elements are float type
    sim_df = sim_df.astype(float)
    # Set index names
    sim_df.index.names = ['sim_id', 'time_id']

    if (fpath is not None) and (fname is not None):
        sim_df.to_excel(_os.path.join(fpath, fname + '.xls'))

    if verbose > 0: print('Simulation completed.')

    return sim_df, ids_list


# -----------------------------------------------------------------------------------------------------------------------
# --------------

DataFrame.rebase = rebase
Series.rebase = rebase
DataFrame.sb_simulate = sb_simulate
DataFrame.ret = full_return_DF
DataFrame.mrets = mrets
DataFrame.stats = stats
Series._rets_stats = _rets_stats
DataFrame.index_freq = index_freq
Series.index_freq = index_freq
DataFrame.cagr = cagr
Series.cagr = cagr
Series.mdd = maxdrawdown
DataFrame.mdd = maxdrawdown
DataFrame.rets2lvl = rets2lvl
Series.rets2lvl = rets2lvl
DataFrame.perf_since = perf_since
DataFrame.add_subgroups = add_subgroups
DataFrame.portfolio = portfolio
DataFrame.portfolio_rets = portfolio_rets
DataFrame.rets2ts = rets2ts
DataFrame.rets2py = rets2py
DataFrame.rolling_ols = rolling_ols
DataFrame.rets2db = rets2db
DataFrame.perf_since = perf_since
DataFrame.mr2ts = mr2ts
DataFrame.shr = sharpe_ratio
DataFrame.convert_timezone = convert_timezone
