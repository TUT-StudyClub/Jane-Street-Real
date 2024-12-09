"""
=========================
Data loading and import
=========================
"""
import kagglehub
kagglehub.login()

jane_street_real_time_market_data_forecasting_path = kagglehub.competition_download('jane-street-real-time-market-data-forecasting')
voix97_js_with_lags_trained_xgb_path = kagglehub.dataset_download('voix97/js-with-lags-trained-xgb')
voix97_js_xs_nn_trained_model_path = kagglehub.dataset_download('voix97/js-xs-nn-trained-model')
xuanleekaggle_jane_street_5_and_7__other_default_1_path = kagglehub.model_download('xuanleekaggle/jane-street-5-and-7_/Other/default/1')

print('Data source import complete.')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv, pd.read_parquet )
import polars as pl

import os, gc
from tqdm.auto import tqdm
import pickle # module to serialize and deserialize objects

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import r2_score
from sklearn.ensemble import VotingRegressor

import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

import kaggle_evaluation.jane_street_inference_server

gridColor = 'lightgrey'