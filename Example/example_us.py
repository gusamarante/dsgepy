"""
This example is not finished
"""

import numpy as np
import pandas as pd
from dsgepy import DSGE
import matplotlib.pyplot as plt
from sympy import Matrix, symbols
from statsmodels.tsa.filters.hp_filter import hpfilter

from dsgepy import FRED

fred = FRED()

# ===== Grab and organize Data ===== #
# Get data from the FRED
series_dict = {'CPIAUCSL': 'CPI',
               'GDP': 'GDP',
               'DFF': 'Fed Funds Rate'}

df = fred.fetch(series_id=series_dict)

# Observed varibles
df_obs = pd.DataFrame()
df_obs['CPI'] = df['CPI'].dropna().resample('Q').last().pct_change(1) * 4
df_obs['FFR'] = df['Fed Funds Rate'].resample('Q').mean()
df_obs['outputgap'], _ = hpfilter(np.log(df['GDP'].resample('Q').last().dropna()), 1600)

df_obs = df_obs.dropna()

df_obs = df_obs[df_obs.index >= '2000-01-01']

# ================================
# ===== MODEL ESPECIFICATION =====
# ================================
# endogenous variables at t


# endogenous variables at t - 1

# exogenous shocks


# expectational shocks


# parameters


# Summary parameters


# model (state) equations


# observation equations


# =============================
# ===== MODEL ESTIMATION  =====
# =============================
# Not all parameters need to be estimated, these are going to be calibrated

# priors


# DSGE object

# Estimation

# Posterior Table

# IRFs from the estimated Model

# Extraxct state variables

# Historical Decomposition
