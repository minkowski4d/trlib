#!/usr/bin/env python
# -*- coding: UTF-8 -*-


# Import Python Modules
import numpy as np
from datetime import datetime as _datetime, date

# Custom Modules
from trlib import pandas_patched as pd


def _get_stockperks_prices(filename=None):
    """
    FB 20220429: Obsolete
    First price for stock perks.
    """
    if filename is None:
        filename = '/Volumes/GoogleDrive/My\ Drive/Risk/StockPerks/RiskReport_20220331/stockperks_instr_returns_etfs_stocks.csv'
        filename = filename.replace("\/", "").replace("\\", "")
        print("\n ***************** Reading data *****************\n")
        print("Opening file: %s"%filename.split('/')[-1])

    # Read the Data
    df = pd.read_csv(filename)
    # Format Output
    # Column Names
    df = df.rename(columns=lambda x: x.lower())
    # Date Object and Index
    df['ddate'] = df['date'].apply(lambda x: _datetime.strptime(x, '%Y-%m-%d'))
    df = df.drop('date', axis=1)
    df = df[['ddate', 'instrument_id', 'name_short', 'instrument_type', 'price']]
    df = df.set_index('ddate')

    # Create Output
    out = dict()
    # Price DataFrame
    out['prices'] = df.reset_index().pivot_table(values='price', index='ddate', columns='instrument_id')
    # Asset Information (isin, name, asset_class):
    asset_info = df.reset_index()[['instrument_id', 'name_short', 'instrument_type']].drop_duplicates(subset=['instrument_id'])
    out['asset_info'] = asset_info.reset_index().drop('index', axis=1)

    return out



def get_yahoo_prices(symbols=['LTC-USD'], field='close', startdate=date(2020, 1, 1), enddate=date(2022, 3, 31), verbose=True):
    """
    Quick code to retrieve cypto price data from yahoo via pandas data_reader
    @param symbols: list,
                    e.g. ['BTC-EUR']
                    for currencies check availability on https://finance.yahoo.com/currencies
                    e.g.  ['EURGBP=X']
    @param field: ohlc, volume or adj_close
    @param startdate: date object
    @param enddate: date object
    """

    import pandas_datareader as web

    out = pd.DataFrame(index = pd.date_range(start = startdate, end = enddate, freq='B'))

    for sym in symbols:
        if verbose: print('Retrieving price data for: %s'%sym)
        try:
            tmp = web.DataReader(sym, 'yahoo', startdate, enddate)
            tmp = tmp.rename(columns = lambda x: x.replace(' ', '_').lower()).rename(columns = {field:sym})
            out = out.join(tmp[[sym]])
        except:
            if verbose: print('Error: Could not retrieve data for %s'%sym)
            pass

    return out



def find_beta_series(df, ids2corr =[]):


    # Import Python Modules
    import seaborn as sns
    import missingno as mno
    from sklearn import linear_model

    if len(ids2corr) == 0:
        # MSCI World ETFs with best data covorage as of 20220509:
        #
        # INSTRUMENT_ID	NAME_SHORT	                INSTRUMENT_TYPE	    CURRENCY
        # IE00B4L5Y983	Core MSCI World USD (Acc)	FUND	            USD
        # IE00B0M62Q58	MSCI World USD (Dist)	    FUND	            USD

        ids2corr = ['IE00B0M62Q58','IE00B4L5Y983']


def get_future_returns():
    """
    Reads major market index returns. Used solely for backtesting
    """
    rets = pd.read_pickle('/Users/fabioballoni/Work/Risk/Projects/FractionalTrading/VaR/Securities/rets_securities.pkl')

    return rets