#!/usr/bin/env python
# -*- coding: UTF-8 -*-


# Import Python Modules
import numpy as np
from datetime import datetime as _datetime, date

# Custom Modules
from trlib import pandas_patched as pd


def get_universe(filename=None, project='caracalla'):

    if filename is None and project == 'caracalla':
        filename = '/Volumes/GoogleDrive/My\ Drive/Risk/Data/caracalla_universe.csv'
        filename = filename.replace("\/", "").replace("\\", "")
        print("\n ***************** Reading data *****************\n")
        print("Project %s - Opening file: %s"%(project,filename.split('/')[-1]))

        df = pd.read_csv(filename)
        df = df.set_index('INSTRUMENT_ID')

    return df


def enrich_data(df, field='currency', verbose=True):


    print('******** Adjusting String Fields ********')
    print('\t * Filling nans in NAME_SHORT with "Unknown"')
    df['NAME_SHORT'].loc[df.NAME_SHORT.isnull()] = 'Unknown'

    if field == 'currency':
        print('\n\t * Starting mapping for fields "CURRENCY" and "COUNTRY"')
        if verbose: print("\t\tRetrieving Data from investpy")
        import investpy
        n = 0
        n_err = 0
        # Create columns:
        df['CURRENCY'] = np.nan
        for k in df.index:
            if len(df[:k]) % 250 == 0:
                n += 250
                print("\t\t\tPassed %sth iteration. Remaining %s"%(n, len(df)-n))
            try:
                tmp = investpy.stocks.search_stocks(by = 'isin', value = k)
                df.loc[k, 'CURRENCY'] = tmp.iloc[0].currency
            except:
                # Additional mapping sequence throughout currencies in the instrument's names
                if 'usd' in df.loc[k, 'NAME_SHORT'].lower():
                    df.loc[k, 'CURRENCY'] = 'USD'
                elif 'eur' in df.loc[k, 'NAME_SHORT'].lower():
                    df.loc[k, 'CURRENCY'] = 'EUR'
                else:
                    n_err += 1
                pass

    if verbose:
        print('\n ******** Error Report ********')
        print('\t Missing currencies: %s'%len(df[df.CURRENCY.isnull()]))

    return df

