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


# def risk_model_test(rets, rescale, plot=True):
#     """
#     Performs various statistical test of gjr and egarch models
#     :param rets:
#     :param rescale:
#     :param plot:
#     :return:
#     """
#
#     from hurst import compute_Hc
#     from statsmodels.stats.diagnostic import acorr_ljungbox
#
#     # Create OutPut:
#     out_dict = dict()
#
#     # Format Data:
#     rets_rescaled = rets.copy() * rescale
#     variance_rescaled = rets_rescaled.sub(rets_rescaled.mean()).pow(2)
#
#     # Calculate Hurst Exponent
#     # https://en.wikipedia.org/wiki/Hurst_exponent#Rescaled_range_(R/S)_analysis
#     # https://pypi.org/project/hurst/
#     # H = 0.5 — Brownian motion,
#     # 0.5 < H < 1.0 — persistent behavior,
#     # 0 < H < 0.5 — anti-persistent behavior.
#
#     out_dict['hurst'], out_dict['hurst_const'], out_dict['hurst_data'] = compute_Hc(rets, kind='change',
#                                                                                     simplified=True)
#     if plot:
#         out_dict['hurst_fig'] = CH.risk_hurst_exponent(out_dict['hurst'], out_dict['hurst_const'],
#                                                        out_dict['hurst_data'])
#         out_dict['correl_fig'] = CH.risk_correlogram(rets, out_dict, lags=100)
#
#     # ******************************************************************************************************
#     # Compare Egarch and GJR Garch Models
#     # Specify GJR-GARCH model assumptions
#     gjr_gm = arch_model(rets_rescaled, p=1, q=1, o=1, vol='GARCH', dist='t')
#     # Fit the model
#     gjrgm_result = gjr_gm.fit(disp='off')
#     # Get Conditional Volatility
#     gjrgm_vol = gjrgm_result.conditional_volatility
#     # Get Residuals
#     gjrgm_resid = gjrgm_result.resid
#     # Calculate Standardized Residuals:
#     gjrgm_std_resid = gjrgm_resid / gjrgm_vol
#
#     # ******************************************************************************************************
#     # Specify EGARCH model assumptions
#     egarch_gm = arch_model(rets_rescaled, p=1, q=1, o=1, vol='EGARCH', dist='t')
#     # Fit the model
#     egarch_result = egarch_gm.fit(disp='off')
#     # Get Conditional Volatility
#     egarch_vol = egarch_result.conditional_volatility
#     # ToDo: Can gjrgm_vol and egarch_vol can be plotted against rets_rescaled
#     # Get Residuals
#     egarch_resid = egarch_result.resid
#     # Calculate Standardized Residuals:
#     egarch_std_resid = egarch_resid / egarch_vol
#
#     # ******************************************************************************************************
#     # Rolling Forecast Windows
#     # Fixed Window
#     index = rets_rescaled.index
#     start_loc = 0
#     end_loc = np.where(index >= date(2020, 1, 1))[0].min()
#     forecasts = {}
#     for i in range(70):
#         sys.stdout.write('-')
#         sys.stdout.flush()
#         res = gjr_gm.fit(first_obs=start_loc + i, last_obs=i + end_loc, disp='off')
#         temp = res.forecast(horizon=1, reindex = False).variance
#         fcast = temp.iloc[i + end_loc - 1]
#         forecasts[fcast.name] = fcast
#     print(' Done!')
#     variance_fixedwin = pd.DataFrame(forecasts).T
#
#     # Expanding Window
#     forecasts = {}
#     for i in range(70):
#         sys.stdout.write('-')
#         sys.stdout.flush()
#         res = gjr_gm.fit(first_obs=start_loc, last_obs=i + end_loc, disp='off')
#         temp = res.forecast(horizon=1, reindex = False).variance
#         fcast = temp.iloc[i + end_loc - 1]
#         forecasts[fcast.name] = fcast
#     print(' Done!')
#
#     variance_expandwin = pd.DataFrame(forecasts).T
#
#     # Comparison
#     # Calculate volatility from variance forecast with an expanding window
#     vol_expandwin = np.sqrt(variance_expandwin)
#     # Calculate volatility from variance forecast with a fixed rolling window
#     vol_fixedwin = np.sqrt(variance_fixedwin)
#     # ToDo: Plot against Rets or rets rescaled
#
#     # ******************************************************************************************************
#     # Simplification with p-Values
#     # Get parameter stats from model summary
#     gjr_para_summary = pd.DataFrame({'parameter': gjrgm_result.params,
#                                      'p-value': gjrgm_result.pvalues,
#                                      'std-err': gjrgm_result.std_err,
#                                      't-value': gjrgm_result.tvalues})
#
#     egarch_para_summary = pd.DataFrame({'parameter': egarch_result.params,
#                                         'p-value': egarch_result.pvalues,
#                                         'std-err': egarch_result.std_err,
#                                         't-value': egarch_result.tvalues})
#
#     # Import the Python module
#
#     # ******************************************************************************************************
#     # Ljung-Box Test
#     # Perform the Ljung-Box test
#     qljungbox_gjr, pval_gjr, qboxpierce_gjr, pvalbp_gjr = acorr_ljungbox(gjrgm_std_resid, lags=10)
#     qljungbox_egarch, pval_egarch, qboxpierce_egarch, pvalbp_egarch = acorr_ljungbox(egarch_std_resid, lags=10)
#
#     # Store p-values in DataFrame
#     df_lb_test = pd.DataFrame({'P-values - GJR Garch': pval_gjr, 'P-values - EGarch': pval_egarch}).T
#     # Create column names for each lag
#     col_num = df_lb_test.shape[1]
#     col_names = ['lag_' + str(num) for num in list(range(1, col_num + 1, 1))]
#     # Display the p-values
#     df_lb_test.columns = col_names
#
#     # Backtesting with MAE, MSE
#
#     def evaluate(observation, forecast):
#         """
#         Runs Observations through MAE and MSE test
#         :param observation:
#         :param forecast:
#         :return:
#         """
#         from sklearn.metrics import mean_absolute_error, mean_squared_error
#         # Call sklearn function to calculate MAE
#         mae = mean_absolute_error(observation, forecast)
#         print('Mean Absolute Error (MAE): {round(mae, 3)}')
#         # Call sklearn function to calculate MSE
#         mse = mean_squared_error(observation, forecast)
#         print('Mean Squared Error (MSE): {round(mse, 3)}')
#         return mae, mse
#
#     # Backtest model with MAE, MSE
#     evaluate(rets_rescaled.iloc[:, 0].sub(rets_rescaled.iloc[:, 0].mean()).pow(2) * 100, egarch_vol ** 2)
#
#     # ToDO: Plot Daily Vol agaistn garch vol
#     if plot:
#         # Plot the actual Bitcoin volatility
#         CH.plot(rets_rescaled.iloc[:, 0].sub(rets_rescaled.iloc[:, 0].mean()).pow(2), color='grey', alpha=0.4,
#                 label='Daily Volatility')
#
#         # Plot EGARCH  estimated volatility
#         CH.plot(egarch_vol ** 2, color='red', label='EGARCH Volatility')
#
#         CH.legend(loc='upper right')
#         CH.show()
#
#     # ******************************************************************************************************
#     # The paths for the final observation
#     sim_forecasts = egarch_result.forecast(horizon=5, method='simulation')
#     sim_paths = sim_forecasts.simulations.residual_variances[-1].T
#     sim = sim_forecasts.simulations
#
#     bs_forecasts = egarch_result.forecast(horizon=5, method='bootstrap')
#     bs_paths = bs_forecasts.simulations.residual_variances[-1].T
#     bs = bs_forecasts.simulations
#
#     if plot:
#         fig, axes = CH.subplots(1, 2, figsize=(13, 5))
#
#         x = np.arange(1, 6)
#
#         # Plot the paths and the mean, set the axis to have the same limit
#         axes[0].plot(x, np.sqrt(252 * sim_paths), color='tomato', alpha=0.2)
#         axes[0].plot(x, np.sqrt(252 * sim_forecasts.residual_variance.iloc[-1]),
#                      color='k', alpha=1)
#
#         axes[0].set_title('Model-based Simulation')
#         axes[0].set_xticks(np.arange(1, 6))
#         axes[0].set_xlim(1, 5)
#
#         axes[1].plot(x, np.sqrt(252 * bs_paths), color='deepskyblue', alpha=0.2)
#         axes[1].plot(x, np.sqrt(252 * bs_forecasts.residual_variance.iloc[-1]),
#                      color='k', alpha=1)
#
#         axes[1].set_xticks(np.arange(1, 6))
#         axes[1].set_xlim(1, 5)
#
#         axes[1].set_title('Bootstrap Scenario')
#         CH.show()
#
#         # Plot Simulation Variances
#         fig, axes = CH.subplots(1, 2, figsize=(13, 5))
#
#         sns.boxplot(data=sim.variances[-1], ax=axes[0])
#         sns.boxplot(data=bs.variances[-1], ax=axes[1])
#
#         axes[0].set_title('Model-based Simulation Variances')
#         axes[1].set_title('Bootstrap Simulation Variances')
#
#         CH.show()
#
#     return out_dict


def calculate_vaR_garch(rets, window=250, qtl=0.01, horizon=1, rescale=100, decay=1, ftol=1e-06, max_iter=150, fhs=False, garch_engine='gjr'):
    """
    Calculate Parametric Value at Risk Quantiles 99&95 throughout GARCH Volatiltiy Model
    :param rets: Single Column percentage change dataframe
    :param window: lookback window, e.g. 250
    :param qtl: quantile, e.g. 0.01 (1 - confidence interval)
    :param horizon: forecast horizon, e.g. 1
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
    rets_garch = rets[-window:].copy()
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
    forecasts = res.forecast(horizon=horizon, start=str(rets_garch.index[-1])[:10], simulations=1000, reindex = False)
    cond_mean = forecasts.mean[-horizon:]
    cond_var = forecasts.variance[-horizon:]
    if fhs:
        std_rets = (rets_garch.iloc[:, 0] - res.params["mu"]).div(res.conditional_volatility)
        std_rets = std_rets.dropna()
        q = np.array(std_rets.quantile([qtl]))
    else:
        # q = am.distribution.ppf([qtl], res.params[-2 if garch_engine == 'egarch' else -1:]) FB20220511 only to use if distr for egarch is 'skewt'
        q = am.distribution.ppf([qtl], res.params[-1:])
    # Multiply Result by -1 fpr better intuition:
    value_at_risk = -1 * (-cond_mean.values - np.sqrt(cond_var).values * q[None, :])
    value_at_risk = pd.DataFrame(value_at_risk, columns=['var%s_1d' %str(int(100 * (1-qtl)))], index=['VaR'])


    return value_at_risk / rescale


