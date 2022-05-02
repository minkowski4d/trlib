#!/usr/bin/env python
# -*- coding: UTF-8 -*-


# Import Python Modules
import numpy as np
import os
from datetime import datetime as _datetime, date

# Custom Modules
from trlib import pandas_patched as pd
from trlib import config as cf


def cache_load(verbose=True):

    # Define Variables:
    data_pth = '/Volumes/GoogleDrive/My Drive/Risk/Data/'

    if verbose: print("\n ***************** Pulling Securities Price Cache *****************\n")
    cf.cache_prices = pd.read_pickle(os.path.join(data_pth, 'cache_prices.pkl'))
    if verbose: print("\n ***************** Price Cache Loaded *****************\n")

    if verbose: print("\n ***************** Pulling Securities Info Cache for Caracalla Universe *****************\n")
    cf.cache_info['caracalla'] = pd.read_pickle(os.path.join(data_pth, 'cache_universe_caracalla.pkl'))
    if verbose: print("\n ***************** Carcalla Universe Loaded *****************\n")



def build_info_cache(save_cache=True, verbose=True):

    from trlib.instruments import data_info as dtf

    if verbose: print("\n ***************** Building Securities Universe Cache *****************\n")
    out_dict = dict()

    # Build Cache for Caracalla universe
    if verbose: print("\n ********* 1. Caracalla Universe Cache *********\n")
    # Check for already dumped cache files
    data_raw_pth = '/Volumes/GoogleDrive/My Drive/Risk/Data/'
    if 'cache_universe_caracalla.pkl' in os.listdir(data_raw_pth):
        caracalla_univ_0 = pd.read_pickle(os.path.join(data_raw_pth, 'cache_universe_caracalla.pkl'))

    # Read New Snowflake Data
    df_new = dtf.get_universe()
    if verbose:
        print("\n\t 1.1 Comparing new with stored data:\n")

    # Compare the datasets
    if df_new[~df_new.index.isin(caracalla_univ_0.index)].empty:
        if verbose: print("\n\t 1.1 Comparing new with stored data:\n")
    else:
        tmp = df_new[~df_new.index.isin(caracalla_univ_0.index)]
        if verbose: print("\n\t 1.1 Fetching data for new %s entries\n"%len(tmp))
        df_enriched = dtf.enrich_data(tmp)
        caracalla_univ_0 = caracalla_univ_0.append(df_enriched)

        if save_cache:
            if verbose: print("\n\t 1.2 Storing new cache for caracalla universe")
            data_pth = '/Volumes/GoogleDrive/My Drive/Risk/Data/'
            caracalla_univ_0.to_pickle(os.path.join(data_pth, 'cache_universe_caracalla.pkl'))

    out_dict['caracalla_info'] = caracalla_univ_0

    cf.cache_info = out_dict



def build_price_cache(create_new=True, save_cache=True,verbose=True):

    from trlib.instruments import data_prices as dtp
    from trlib.instruments import data_info as dtf

    if verbose: print("\n ***************** Building Securities Price Cache *****************\n")

    # Build Cache for Prices
    if verbose: print("\t 1. Check for already saved cache")
    # Check for already dumped cache files
    data_pth = '/Volumes/GoogleDrive/My Drive/Risk/Data/'
    data_raw_pth = '/Volumes/GoogleDrive/My Drive/Risk/Data/DataRaw/'
    if 'cache_prices.pkl' in os.listdir(data_pth):
        prices_univ_0 = pd.read_pickle(os.path.join(data_raw_pth,'cache_prices_raw.pkl'))
        if verbose: print("\t\t Found cache_prices with oldest entry as of %s"%max(prices_univ_0.PRICE_DT.unique()))
    else:
        raise Exception('Warning: Could not find any saved price file')
        sys.exit()

    # Read New Snowflake Data
    df_new = pd.read_csv(os.path.join(data_raw_pth, 'histr_returns_10D.csv'))
    df_new.columns = ['PRICE_DT','INSTRUMENT_ID','NAME_SHORT','INSTRUMENT_TYPE','PRICE']
    df_new = df_new[['PRICE_DT', 'INSTRUMENT_ID', 'PRICE']]

    if verbose:
        print("\n\t 2. Comparing new with stored data:")

    # Compare the datasets
    if verbose: print("\t\t Concatenating old with new entries")
    price_univ_1 = pd.concat([prices_univ_0, df_new], axis=0)
    price_univ_1 = price_univ_1.drop_duplicates(subset = ['PRICE_DT','INSTRUMENT_ID'])

    if verbose: print("\t\t Storing new raw historical price in histr_returns.pkl")
    price_univ_1.to_pickle(os.path.join(data_raw_pth,'cache_prices_raw.pkl'))

    if verbose: print("\t\t Found %s new prices. \n"%((len(prices_univ_0)+len(df_new)) - len(price_univ_1)))

    # Unstacking Data into DataFrame - columns -> ISINs, index -> date object
    df_prx = price_univ_1.pivot_table(index = price_univ_1.PRICE_DT,columns = price_univ_1.INSTRUMENT_ID,values = 'PRICE')
    df_prx = df_prx.dropna(how = 'all',axis = 0)
    df_prx = df_prx.rename_axis(None, axis = 1)
    df_prx = df_prx.rename(index = lambda x: _datetime.strptime(x,"%Y-%m-%d"))


    if save_cache:
        if verbose: print("\n\t 3. Storing new cache for prices")
        df_prx.to_pickle(os.path.join(data_pth, 'cache_prices.pkl'))
        if verbose: print("\n\t 4. Storing new cache raw for prices")


