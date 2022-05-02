#!/usr/bin/env python
# -*- coding: UTF-8 -*-


# Import Python Modules
import numpy as np
from datetime import datetime as _datetime, date

# Custom Modules
from trlib import pandas_patched as pd


def clean_prices(df, verbose=True):
    """
    Price Data Cleaner. Input Dictionary should contain a key "prices" and that DataFrame should be be built as:
    index -  date object
    columns - ISINs
    values - prices
    """

    # Output
    out_dict = dict()

    # **********************************************************************************
    # Check for NaNs
    df_nans = df.copy()
    df_nans[df_nans.notna()] = 0
    df_nans[~df_nans.notna()] = 1

    # Filter out first only NaN entries as those shouldn't be counted
    for col in df_nans.columns:
        df_nans[col].loc[:df[col].first_valid_index()] = 0


    if verbose:
        # Print Results:
        topic_count = 1
        print('------------------------------------------------\n')
        if len(df_nans.columns) <= 10:
            print('%s. Missing Prices for these securities:\n'%topic_count)
            print(f"\t{'Symbol':16} {'Date':12}")
            for k_nan in df_nans.columns:
                tmp = df_nans[k_nan]
                tmp = tmp[tmp.isnull().values]
                count = 0
                for j in tmp.index:
                    if count > 0:
                        print(f"\t{'':16} {str(j):12}")
                    else:
                        print(f"\t{k_nan:16} {str(j):12}")
                    count += 1

            print('\nSolution: front filling with prices as of T-1')
        else:
            print('%s. Missing Prices exceed console printing capabilities. Check output\n'%topic_count)
        topic_count += 1
        print('\n------------------------------------------------\n')

    # Filling Missing Prices
    # ToDO: Check how to fill! first index?
    df = df.fillna(method = 'ffill')

    out_dict['prx'] = df

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

    out_dict['rets'] = rets
    print('\n------------------------------------------------\n')

    return out_dict

