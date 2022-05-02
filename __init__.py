#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import division
import sys as _sys
import platform as _platform
import getpass as _getpass
from multiprocessing import current_process
from trlib import pandas_patched as pd
from datetime import datetime as _datetime
from datetime import date
from pdb import pm as postm
from pdb import runcall as dbg
from pdb import runeval as dbge
from importlib import reload

platform = _platform.system()
user_node = _platform.node()
user_name = _getpass.getuser()

if current_process().name == 'MainProcess':
    print("")
    print("     ------------------------------------------------------------")
    print("               Risk @ Trade Republic Bank GmbH")
    print("     ------------------------------------------------------------")
    print('      Node: %s\tUser: %s'%(user_node, user_name))
    print('      Python version '+_sys.version.replace('\n', '\n\t'))
    print('      pandas: %s\tnumpy: %s'%(pd.__version__, pd.np.__version__))


if user_name.startswith('fabioballoni'):

    # Import Python MoDules
    import numpy as np
    import matplotlib
    import seaborn as sns
    import sys
    import os
    from datetime import datetime as _datetime, date
    sns.set()

    # Custom Modules
    # Projects
    from trlib.stock_perks import sp_data as spd

    # Instruments
    from trlib.instruments import data_prices as dtp
    from trlib.instruments import data_info as dtf
    from trlib.instruments.data_info import sid

    # Mixed
    from trlib import cache_data as cad
    from trlib.cache_data import cache_load
    from trlib import config as cf

    # Matplotlib Parameters
    matplotlib.rcParams['figure.facecolor'] = '1'
    matplotlib.rcParams["axes.facecolor"] = '1'
    matplotlib.rcParams["axes.edgecolor"] = '0.75'
    matplotlib.rcParams['grid.color'] = '0.75'
    matplotlib.rcParams['grid.linestyle'] = ':'
    matplotlib.rcParams['grid.linewidth'] = 0.5
    matplotlib.rcParams['axes.xmargin'] = 0
    matplotlib.rcParams['legend.shadow'] = True
    matplotlib.rcParams['legend.framealpha'] = 1
    matplotlib.rcParams['legend.loc'] = "upper left"
    pd.set_option('display.float_format',lambda x: '%0.6f'%x)
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.max_colwidth', 40)
    pd.set_option('display.max_rows', 40)
    pd.set_option('display.width', None)
    pd.options.mode.chained_assignment = None


if current_process().name == 'MainProcess':
    print('      Kernel started '+_datetime.now().strftime('%Y-%m-%d %H:%M'))
    print("     ------------------------------------------------------------")
