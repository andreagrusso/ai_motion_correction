# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:18:03 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb
import os


experiment_id = 'rVS0BinSQ9Gcj9yfB7nLVA'
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
df = experiment.get_scalars(pivot=True)



df_validation = 
#sns.lineplot(data=df,x=)