#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# flake8: noqa

# In[]:
# Controls for webapp


# Python Modules
from datetime import datetime, timedelta
import pickle
import numpy as np
import traceback


# Caracalla ***************************************************************************************

# Data Quality
def caching_data(df, verbose=True):

    from trlib.instruments import data_quality as dtq