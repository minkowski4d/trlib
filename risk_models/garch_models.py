#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Python Modules
import sys
import numpy as np
from arch import arch_model
import seaborn as sns
from datetime import datetime as _datetime, date, timedelta


# QR Modules
from trlib import pandas_patched as pd



def garch_exp(rets_rescaled, dist='t'):
    """
    Calculation of  EGarch Volatility and Residual Returns
    :param rets_rescaled: returns series multiplied by rescaling factor, e.g. 100
    :param dist: distirbution that is used for maximum likelyhood estimation
    """

    # Specify EGARCH model assumptions
    egarch_gm = arch_model(rets_rescaled, p=1, q=1, o=1, vol='EGARCH', dist=dist)
    # Fit the model
    egarch_result = egarch_gm.fit(disp='off')
    # Get Conditional Volatility
    egarch_vol = egarch_result.conditional_volatility

    # Get Residuals
    egarch_resid = egarch_result.resid
    # Calculate Standardized Residuals:
    egarch_std_resid = egarch_resid / egarch_vol

    return egarch_gm, egarch_result, egarch_vol, egarch_resid, egarch_std_resid


def garch_gjr(rets_rescaled, dist='t'):
    """
    Calculation of  GJR Volatility and Residual Returns
    :param rets_rescaled: returns series multiplied by rescaling factor, e.g. 100
    :param dist: distribution, default 't'. Options: 'norm', 'skewt'
    """
    gjr_gm = arch_model(rets_rescaled, p=1, q=1, o=1, vol='GARCH', dist=dist)
    # Fit the model
    gjrgm_result = gjr_gm.fit(disp='off')
    # Get Conditional Volatility
    gjrgm_vol = gjrgm_result.conditional_volatility
    # Get Residuals
    gjrgm_resid = gjrgm_result.resid
    # Calculate Standardized Residuals:
    gjrgm_std_resid = gjrgm_resid / gjrgm_vol

    return gjr_gm, gjrgm_result, gjrgm_vol, gjrgm_resid, gjrgm_std_resid



def calculate_vaR_garch(rets, qtl=0.01, holding_period=1, rescale=100, decay=1, ftol=1e-06, max_iter=150, fhs=False, garch_engine='gjr'):
    """
    Calculate Parametric Value at Risk Quantiles 99&95 throughout GARCH Volatiltiy Model
    :param rets: Single Column percentage change dataframe
    :param qtl: quantile, e.g. 0.01 (1 - confidence interval)
    :param holding_period: forecast horizon, e.g. 1
    :param rescale: e.g. 100
    :param decay: applied via an exponentially weighted moving average, e.g. 0.94 or 1 for no decay
    :param ftol: Precision goal for the value of f in the stopping criterion in slsqp optimizer, e.g. 1e-03
    :param max_iter: iteration limit for the optimizer to reach convergence
    :return: dataframe with VaR Values for 95,99 and 1D,20D
    """

    # Adjust Index format
    if rets.index.dtype == 'O':
        rets = rets.rename(index=lambda x: _datetime(x.year, x.month, x.day))

    # Apply Sample Size
    rets_garch = rets.copy()
    # Use decay
    if decay != 1:
        rets_garch = rets_garch.sort_index(ascending=False).ewm(alpha=decay, adjust=True).mean()

    # Run Garch Model
    rets_garch = rets_garch.sort_index() * rescale

    if garch_engine == 'egarch':
        am = garch_exp(rets_garch)[0]
    elif garch_engine == 'gjr':
        am = garch_gjr(rets_garch)[0]

    res = am.fit(disp='off', last_obs=str(rets_garch.index[-1])[:10], options={'maxiter': max_iter, 'ftol': ftol})
    forecasts = res.forecast(horizon=holding_period, start=str(rets_garch.index[-1])[:10], simulations=1000, reindex = False)
    cond_mean = forecasts.mean[-holding_period:]
    cond_var = forecasts.variance[-holding_period:]
    if fhs:
        std_rets = (rets_garch.iloc[:, 0] - res.params["mu"]).div(res.conditional_volatility)
        std_rets = std_rets.dropna()
        q = np.array(std_rets.quantile([qtl]))
    else:
        # q = am.distribution.ppf([qtl], res.params[-2 if garch_engine == 'egarch' else -1:]) FB20220511 only to use if distr for egarch is 'skewt'
        q = am.distribution.ppf([qtl], res.params[-1:])
    # Multiply Result by -1 fpr better intuition:
    value_at_risk = -1 * cond_mean.values + np.sqrt(cond_var).values * q[None, :]
    value_at_risk = pd.DataFrame(value_at_risk, columns=['var%s_1d' %str(int(100 * (1-qtl)))], index=['VaR'])


    return value_at_risk / rescale


