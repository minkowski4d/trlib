#!/usr/bin/env python
# -*- coding: UTF-8 -*-



# Import Python Modules
import numpy as np


# Custom Modules
from trlib import pandas_patched as pd
from trlib import utils as ut
from scipy.stats import multivariate_t, multivariate_normal



def ewma_volatility(rets, window=250, decay=1):
    """
    Calculates EWMA Volatility on a given return series:
    https://www.une.edu.au/__data/assets/pdf_file/0009/76464/unebsop14-1.pdf
    """
    out = pd.DataFrame(index = rets[window-1:].index, columns = rets.columns)

    # Calculate EWM weights
    # SUM [lambda^(t-1) * (1 - lambda)]
    ewm_values = np.power(decay, np.arange(window - 1, -1, -1))*(1 - decay)

    for idx in rets[window-1:].index:
        # Slice to lookback window
        tmp = rets[:idx][-window:]
        # Assign Standard Deviation Forecast
        # SUM [lambda^(t-1) * (1 - lambda)] * R_t^2
        out.loc[idx] = np.sqrt(np.sum(pd.DataFrame(ewm_values).values * tmp**2))

    return out
