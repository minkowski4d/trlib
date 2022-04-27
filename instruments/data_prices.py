#!/usr/bin/env python
# -*- coding: UTF-8 -*-


# Import Python Modules
import numpy as np
from datetime import datetime as _datetime, date

# Custom Modules
from trlib import pandas_patched as pd


def get_sf_prices(filename=None):

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



def clean_sf_prices(df, verbose=True):
    """
    Price Data Cleaner. Input Dictionary should contain a key "prices" and that DataFrame should be be built as:
    index -  date object
    columns - ISINs
    values - prices

    E.g. retrievable with trlib.data.data_get.get_sf_returns()
    """

    # **********************************************************************************
    # Check for NaNs
    df_nans = df.reindex(df[df.isnull().values].index.drop_duplicates())

    if verbose:
        # Print Results:
        topic_count = 1
        print('------------------------------------------------\n')
        print('%s. Missing Prices for these securities:\n'%topic_count)
        print(f"\t{'Symbol':16} {'Date':12}")
        for k_nan in df_nans.columns:
            tmp = df_nans[k_nan]
            tmp = tmp[tmp.isnull().values]
            count = 0
            for j in tmp.index:
                if count > 0:
                    print(f"\t{'':16} {str(j.date()):12}")
                else:
                    print(f"\t{k_nan:16} {str(j.date()):12}")
                count += 1

        print('\nSolution: front filling with prices as of T-1')
        topic_count += 1
        print('\n------------------------------------------------\n')

    # Filling Missing Prices
    df = df.fillna(method = 'ffill')

    # **********************************************************************************
    # Check for potential corporate actions
    rets = df.pct_change().dropna()
    # Set return threshold
    rets_limit = - 0.4
    rets_corp = rets[rets.values <= rets_limit]
    rets_corp = rets_corp.loc[~rets_corp.index.duplicated(keep='first')]
    rets_corp = rets_corp.loc[:, (rets_corp != 0).any(axis=0)]

    if verbose:
        # Print Results:
        print('%s. Check for potential corporate actions:\n'%topic_count)
        print(f"\t{'Symbol':16} {'Date':12} {'Price Return':12}")
        for k_corp in rets_corp.columns:
            tmp = rets_corp[k_corp]
            tmp = tmp[tmp.values <= rets_limit]
            count = 0
            for j in tmp.index:
                if count > 0:
                    print(f"\t{'':16} {str(j.date()):12} {str(np.round(100*tmp.loc[tmp.index[0]],2))+'%':5}")
                else:
                    print(f"\t{k_nan:16} {str(j.date()):12} {str(np.round(100*tmp.loc[tmp.index[0]],2))+'%':5}")
                count += 1

        print('\nSolution: filling return outliers with Null\n')
        topic_count += 1

    # Setting
    for k_corp in rets_corp.columns:
        for j in rets_corp.index:
            if rets.loc[j, k_corp] <= rets_limit:
                if verbose: print('\tSetting Return to Null for %s at %s'%(k_corp, j))
                rets.loc[j, k_corp] = 0

    print('\n------------------------------------------------\n')

    return df, rets



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

    out = pd.DataFrame(index = pd.date_range(start = startdate, end = enddate))

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



