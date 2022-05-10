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

    out_dict['df_nans'] = df_nans

    # Filter out first only NaN entries as those shouldn't be counted
    # and Fill Missing Prices
    topic_count = 1
    print('**********************************************************\n')
    if verbose: print('%s. Searching for Missing prices\n'%topic_count)
    count = 0
    print("\t\t Reassigning NaN and front filling missing prices on %s columns"%len(df.columns))
    for col in df_nans.columns:
        if verbose:
            if count%500 == 0: print("\t\t\t Screened %s columns"%count)
        # Reassign true NaNs -> Series starts after index[0] or terminates prior index[-1]
        df_nans[col].loc[:df[col].first_valid_index()].iloc[:-1] = 0
        df_nans[col].loc[df[col].last_valid_index():].iloc[1:] = 0

        # Front filling prices
        tmp = df[col].loc[df[col].first_valid_index():df[col].last_valid_index()].fillna(method = 'ffill')
        df[col].loc[df[col].first_valid_index():df[col].last_valid_index()] = tmp
        count += 1


    if verbose:
        # Print Results:
        topic_count += 1
        print('**********************************************************\n')
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
        print('**********************************************************\n')


    out_dict['prx'] = df

    # **********************************************************************************
    # Check for potential corporate actions
    rets = df.pct_change()
    # Set return threshold
    rets_limit = 0.5


    if verbose:
        # Print Results:
        print('%s. Check for potential corporate actions:\n'%topic_count)
        print(f"\t{'Symbol':16} {'Date':12} {'Price Return':12}")
        for k_corp in rets.columns:
            # Skipping Penny Stocks as those tend to have high price changes
            if df[k_corp].mean() >= 1:
                tmp = rets[k_corp].dropna()
                tmp = tmp[(tmp.values <= -1 * rets_limit) | (tmp.values >= rets_limit)]
                if tmp.empty is False:
                    count = 0
                    for j in tmp.index:
                        if count > 0:
                            print(f"\t{'':16} {str(j):12} {str(np.round(100*tmp.loc[tmp.index[0]],2))+'%':5}")
                        else:
                            print(f"\t{k_corp:16} {str(j):12} {str(np.round(100*tmp.loc[tmp.index[0]],2))+'%':5}")

                        # Setting outliers to 0
                        if rets.loc[j, k_corp] <= -1*rets_limit or rets.loc[j, k_corp] >= rets_limit:
                            if verbose: print('\tSetting Return to Null for %s at %s'%(k_corp,j))
                            rets.loc[j, k_corp] = 0
                        count += 1

        print('\nSolution: filling return outliers with Null\n')
        topic_count += 1


    out_dict['rets'] = rets
    print('\n------------------------------------------------\n')

    return out_dict

