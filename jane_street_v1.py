"""
=========================
Data loading and preprocessing
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

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, PercentFormatter

import os, gc
from tqdm.auto import tqdm
import pickle # module to serialize and deserialize objects
import re # for Regular expression operations

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data  import Dataset, DataLoader
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Timer

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor

import lightgbm as lgb
from lightgbm import LGBMRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

import kaggle_evaluation.jane_street_inference_server

gridColor = 'lightgrey'

"""
=========================
Time series analysis and EDA
=========================
"""
train =sample_df
train['N']=train.index.values
train['id']=train.index.values

xx= sample_df[(sample_df.symbol_id==1)] ['id']
yy=sample_df[ (sample_df.symbol_id==1)]['responder_6']

plt.figure(figsize=(16, 5))
plt.plot(xx,yy, color = 'black', linewidth =0.05)
plt.suptitle('Returns, responder_6', weight='bold', fontsize=16)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Returns", fontsize=12)
plt.grid(color = gridColor , linewidth=0.8)
plt.axhline(0, color='red', linestyle='-', linewidth=1.2)
plt.show()

#for symbol_id=1
plt.figure(figsize=(14, 4))
plt.plot(xx,yy.cumsum(), color = 'black', linewidth =0.6)
plt.suptitle('Cumulative responder_6', weight='bold', fontsize=16)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Cumulative res", fontsize=12)
plt.yticks(np.arange(-500,1000,250))
#plt.xticks(np.arange(0,170,10))
plt.grid(color = gridColor)
#plt.grid(color = 'lightblue')
plt.axhline(0, color='red', linestyle='-', linewidth=0.7)
plt.show()

# for symbol_id == 0
plt.figure(figsize=(18, 7))
predictor_cols = [col for col in sample_df.columns if 'responder' in col]
for i in predictor_cols:
    if i == 'responder_6':
        c='red'
        lw=2.5
        plt.plot((sample_df[sample_df.symbol_id == 0].groupby(['date_id'])[i].mean()).cumsum(), linewidth = lw, color = c)
    else:
        lw=1
        plt.plot((sample_df[sample_df.symbol_id == 0].groupby(['date_id'])[i].mean()).cumsum(), linewidth = lw)

plt.xlabel('Trade days')
plt.ylabel('Cumulative response')
plt.title('Response time series over trade days  \n Responder 6 (red) and other responders', weight='bold')
plt.grid(visible=True, color = gridColor, linewidth = 0.7)
plt.axhline(0, color='blue', linestyle='-', linewidth=1)
plt.legend(predictor_cols)
sns.despine()
#plt.show()

plt.figure(figsize=(6, 6))
responders = pd.read_csv(f"{path}/responders.csv")
matrix = responders[[ f"tag_{no}" for no in range(0,5,1) ] ].T.corr()
sns.heatmap(matrix, square=True, cmap="coolwarm", alpha =0.9, vmin=-1, vmax=1, center= 0, linewidths=0.5,
            linecolor='white', annot=True, fmt='.2f')
plt.xlabel("Responder_0 - Responder_8")
plt.ylabel("Responder_0 - Responder_8")
plt.show()

df_train=sample_df
s_id = 0                        # Change params to take a look at other symbols
res_columns = [col for col in df_train.columns if re.match("responder_", col)]
row = 9
j = 0

fig, axs = plt.subplots(figsize=(18, 4*row))
for i in range(1, 3 * len(res_columns) + 1, 3):
    xx= sample_df[(sample_df.symbol_id==s_id)] ['N']
    yy=sample_df[ (sample_df.symbol_id==s_id)][f'responder_{j}']
    c='black'
    if j == 6: c='red'

    ax1 = plt.subplot(9, 3, i)
    ax1.plot(   xx,yy.cumsum()   , color = c, linewidth =0.8 )
    plt.axhline(0, color='blue', linestyle='-', linewidth=0.9)
    plt.grid(color =gridColor )

    ax2 = plt.subplot(9, 3, i+1)
    #by_date = df_symbolX.groupby(["date_id"])
    ax2.plot(xx,yy   , color = c, linewidth =0.05)
    plt.axhline(0, color='blue', linestyle='-', linewidth=1.2)
    ax2.set_title(f"responder_{j}", fontsize = 14)
    plt.grid(color = gridColor)

    ax3 = plt.subplot(9, 3, i+2)
    b=1000
    ax3.hist(yy, bins=b, color = c,density=True, histtype="step" )
    ax3.hist(yy, bins=b, color = 'lightgrey',density=True)
    plt.grid(color = gridColor)
    ax3.set_ylim([0, 3.5])
    ax3.set_xlim([-2.5, 2.5])

    j = j + 1

fig.patch.set_linewidth(3)
fig.patch.set_edgecolor('#000000')
fig.patch.set_facecolor('#eeeeee')
plt.show()

res_columns = [col for col in df_train.columns if re.match("responder_", col)]
row=10
fig, axs = plt.subplots(figsize=(18, 5*row))
b=300
j = 0
for i in range(1, 3 * row + 1, 3):
    xx= sample_df[(sample_df.symbol_id==j)] ['N']
    yy= sample_df[(sample_df.symbol_id==j)]['responder_6']
    c='black'

    ax1 = plt.subplot(row, 3, i)
    ax1.plot(   xx,yy.cumsum()   , color = c, linewidth =0.8 )
    plt.axhline(0, color='red', linestyle='-', linewidth=0.7)
    plt.grid(color = gridColor)
    plt.xlabel('Time')

    ax2 = plt.subplot(row, 3, i+1)
    ax2.plot(xx,yy   , color = c, linewidth =0.05)
    plt.axhline(0, color='red', linestyle='-', linewidth=0.7)
    ax2.set_title(f"symbol_id={j}", fontsize = '14')
    plt.grid(color = gridColor)
    plt.xlabel('Time')

    ax3 = plt.subplot(row, 3, i+2)
    ax3.hist(yy, bins=b, color = c, density=True, histtype="step" )
    ax3.hist(yy, bins=b, color = 'lightgrey',density=True)
    plt.grid(color = gridColor)
    ax3.set_xlim([-2.5, 2.5])
    ax3.set_ylim([0, 1.5])
    plt.xlabel('Time')

    j = j + 1

fig.patch.set_linewidth(3)
fig.patch.set_edgecolor('#000000')
fig.patch.set_facecolor('#eeeeee')
plt.show()

df_train = sample_df
plt.figure(figsize=(20, 3))    # Plot missing values
plt.bar(x=df_train.isna().sum().index, height=df_train.isna().sum().values, color="red", label='missing')   # analog: using missingno
plt.xticks(rotation=90)
plt.title(f'Missing values over the {len(df_train)} samples which have a target')
plt.grid()
plt.legend()
plt.show()

features = pd.read_csv(f"{path}/features.csv")
features

plt.figure(figsize=(18, 6))
plt.imshow(features.iloc[:, 1:].T.values, cmap="gray_r")
plt.xlabel("feature_00 - feature_78")
plt.ylabel("tag_0 - tag_16")
plt.yticks(np.arange(17))
plt.xticks(np.arange(79))
plt.grid(color = 'lightgrey')
plt.show()

plt.figure(figsize=(11, 11))
matrix = features[[ f"tag_{no}" for no in range(0,17,1) ] ].T.corr()
sns.heatmap(matrix, square=True, cmap="coolwarm", alpha =0.9, vmin=-1, vmax=1, center= 0, linewidths=0.5, linecolor='white')
plt.show()


responders = pd.read_csv(f"{path}/responders.csv")
responders


sample_df['weight'].describe().round(1)

plt.figure(figsize=(8,3))
plt.hist(sample_df['weight'], bins=30, color='grey', edgecolor = 'white',density=True )
plt.title('Distribution of weights')
plt.grid(color = 'lightgrey', linewidth=0.5)
plt.axvline(1.7, color='red', linestyle='-', linewidth=0.7)
plt.show()


sub = pd.read_csv(f"{path}/sample_submission.csv")
print( f"shape = {sub.shape}" )
sub.head(10)


col =[]
for i in range(9):
    col.append(f"responder_{i}")

sample_df[col].describe().round(1)

numerical_features=[]
numerical_features=sample_df.filter(regex='^responder_').columns.tolist() # Separate responders
numerical_features.remove('responder_6')

gs=600
k=1;
col = 3
row = 3
fig, axs = plt.subplots(row, col, figsize=(5*col, 5*row))

for i in numerical_features:

    plt.subplot(col,row, k)
    plt.hexbin(sample_df[i], sample_df['responder_6'], gridsize=gs, cmap='CMRmap', bins='log', alpha = 0.2)
    plt.xlabel(f'{i}', fontsize = 12)
    plt.ylabel('responder_6', fontsize = 12)
    plt.tick_params(axis='x', labelsize=6)
    plt.tick_params(axis='y', labelsize=6)
    k=k+1
fig.patch.set_linewidth(3)
fig.patch.set_edgecolor('#000000')
fig.patch.set_facecolor('#eeeeee')

plt.show()

numerical_features=[]
for i in ['05', '06', '07', '08', '12', '15', '19', '32', '38', '39', '50', '51', '65', '66', '67']:
    numerical_features.append(f'feature_{i}')

gs=600
k=1;
col = 3
row = int(np.ceil(len(numerical_features) /3 ))
sz=5
w=sz*col
h = w/col *row
plt.figure(figsize=(w, h))

fig, axs = plt.subplots(figsize=(w, h))

for i in numerical_features:

    plt.subplot(row, col, k)
    plt.hexbin(sample_df['responder_6'], sample_df[i], gridsize=gs, cmap='CMRmap', bins='log', alpha = 0.3)

    plt.xlabel(f'{i}')
    plt.ylabel('responder_6')
    plt.tick_params(axis='x', labelsize=6)
    plt.tick_params(axis='y', labelsize=6)
    k=k+1

fig.patch.set_linewidth(3)
fig.patch.set_edgecolor('#000000')
fig.patch.set_facecolor('#eeeeee')
plt.show()

numerical_features=[]

for i in range(5,9):
    numerical_features.append(f'feature_0{i}')
for i in range(15,20):
    numerical_features.append(f'feature_{i}')

a=0; k=1;
n=3;

fig, axs = plt.subplots(figsize=(15, 4))
for i in numerical_features[:-1]:
    a=a+1
    for j in numerical_features[a:]:
        plt.subplot(1,n, k)
        plt.hexbin(sample_df[i], sample_df[j], gridsize=200, cmap='CMRmap', bins='log', alpha = 1)
        plt.grid()
        plt.xlabel(f'{i}', fontsize = 14)
        plt.ylabel(f'{j}', fontsize = 14)
        plt.tick_params(axis='x', labelsize=6)
        plt.tick_params(axis='y', labelsize=6)

        k=k+1
        if k == (n+1):
            k=1
            plt.show()
            plt.figure(figsize=(15, 4))


"""
=========================
Modelling
=========================
"""
ENSEMBLE_SOLUTIONS = ['SOLUTION_14','SOLUTION_5']
OPTION,__WTS = 'option 91',[0.899, 0.28]

def predict(test:pl.DataFrame, lags:pl.DataFrame | None) -> pl.DataFrame | pd.DataFrame:
    pdB = predict_14(test,lags).to_pandas()
    pdC = predict_5 (test,lags).to_pandas()

    pdB = pdB.rename(columns={'responder_6':'responder_B'})
    pdC = pdC.rename(columns={'responder_6':'responder_C'})
    pds = pd.merge(pdB,pdC, on=['row_id'])
    pds['responder_6'] =\
        pds['responder_B'] *__WTS[0] +\
        pds['responder_C'] *__WTS[1]

    display(pds)
    predictions = test.select('row_id', pl.lit(0.0).alias('responder_6'))
    pred = pds['responder_6'].to_numpy()
    predictions = predictions.with_columns(pl.Series('responder_6', pred.ravel()))
    return predictions


if 'SOLUTION_5' in ENSEMBLE_SOLUTIONS:

    def predict_5(test,lags):
        cols=[f'feature_0{i}' if i<10 else f'feature_{i}' for i in range(79)]
        predictions = test.select(
            'row_id',
            pl.lit(0.0).alias('responder_6'),
        )
        test_preds=model_5.predict(test[cols].to_pandas().fillna(3).values)
        predictions = predictions.with_columns(pl.Series('responder_6', test_preds.ravel()))
        return predictions

if 'SOLUTION_5' in ENSEMBLE_SOLUTIONS:
    from sklearn.linear_model import BayesianRidge
    import joblib
    model_5 = joblib.load('/kaggle/input/jane-street-5-and-7_/other/default/1/ridge_model_5(1).pkl')


if 'SOLUTION_14' in ENSEMBLE_SOLUTIONS:

    class CONFIG:
        seed = 42
        target_col = "responder_6"
        # feature_cols = ["symbol_id", "time_id"] + [f"feature_{idx:02d}" for idx in range(79)]+ [f"responder_{idx}_lag_1" for idx in range(9)]
        feature_cols = [f"feature_{idx:02d}" for idx in range(79)]+ [f"responder_{idx}_lag_1" for idx in range(9)]

        model_paths = [
            #"/kaggle/input/js24-train-gbdt-model-with-lags-singlemodel/result.pkl",
            #"/kaggle/input/js24-trained-gbdt-model/result.pkl",
            "/kaggle/input/js-xs-nn-trained-model",
            "/kaggle/input/js-with-lags-trained-xgb/result.pkl",
        ]

if 'SOLUTION_14' in ENSEMBLE_SOLUTIONS:

    valid = pl.scan_parquet(
        f"/kaggle/input/js24-preprocessing-create-lags/validation.parquet/"
    ).collect().to_pandas()

if 'SOLUTION_14' in ENSEMBLE_SOLUTIONS:

    xgb_model = None
    model_path = CONFIG.model_paths[1]
    with open( model_path, "rb") as fp:
        result = pickle.load(fp)
        xgb_model = result["model"]

    xgb_feature_cols = ["symbol_id", "time_id"] + CONFIG.feature_cols

if 'SOLUTION_14' in ENSEMBLE_SOLUTIONS:

    # Custom R2 metric for validation
    def r2_val(y_true, y_pred, sample_weight):
        r2 = 1 - np.average((y_pred - y_true) ** 2, weights=sample_weight) / (np.average((y_true) ** 2, weights=sample_weight) + 1e-38)
        return r2


    class NN(LightningModule):
        def __init__(self, input_dim, hidden_dims, dropouts, lr, weight_decay):
            super().__init__()
            self.save_hyperparameters()
            layers = []
            in_dim = input_dim
            for i, hidden_dim in enumerate(hidden_dims):
                layers.append(nn.BatchNorm1d(in_dim))
                if i > 0:
                    layers.append(nn.SiLU())
                if i < len(dropouts):
                    layers.append(nn.Dropout(dropouts[i]))
                layers.append(nn.Linear(in_dim, hidden_dim))
                # layers.append(nn.ReLU())
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, 1))
            layers.append(nn.Tanh())
            self.model = nn.Sequential(*layers)
            self.lr = lr
            self.weight_decay = weight_decay
            self.validation_step_outputs = []

        def forward(self, x):
            return 5 * self.model(x).squeeze(-1)

        def training_step(self, batch):
            x, y, w = batch
            y_hat = self(x)
            loss = F.mse_loss(y_hat, y, reduction='none') * w
            loss = loss.mean()
            self.log('train_loss', loss, on_step=False, on_epoch=True, batch_size=x.size(0))
            return loss

        def validation_step(self, batch):
            x, y, w = batch
            y_hat = self(x)
            loss = F.mse_loss(y_hat, y, reduction='none') * w
            loss = loss.mean()
            self.log('val_loss', loss, on_step=False, on_epoch=True, batch_size=x.size(0))
            self.validation_step_outputs.append((y_hat, y, w))
            return loss

        def on_validation_epoch_end(self):
            """Calculate validation WRMSE at the end of the epoch."""
            y = torch.cat([x[1] for x in self.validation_step_outputs]).cpu().numpy()
            if self.trainer.sanity_checking:
                prob = torch.cat([x[0] for x in self.validation_step_outputs]).cpu().numpy()
            else:
                prob = torch.cat([x[0] for x in self.validation_step_outputs]).cpu().numpy()
                weights = torch.cat([x[2] for x in self.validation_step_outputs]).cpu().numpy()
                # r2_val
                val_r_square = r2_val(y, prob, weights)
                self.log("val_r_square", val_r_square, prog_bar=True, on_step=False, on_epoch=True)
            self.validation_step_outputs.clear()

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                                                   verbose=True)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                }
            }

        def on_train_epoch_end(self):
            if self.trainer.sanity_checking:
                return
            epoch = self.trainer.current_epoch
            metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in self.trainer.logged_metrics.items()}
            formatted_metrics = {k: f"{v:.5f}" for k, v in metrics.items()}
            print(f"Epoch {epoch}: {formatted_metrics}")

if 'SOLUTION_14' in ENSEMBLE_SOLUTIONS:

    N_folds = 5
    models = []
    for fold in range(N_folds):
        checkpoint_path = f"{CONFIG.model_paths[0]}/nn_{fold}.model"
        model = NN.load_from_checkpoint(checkpoint_path)
        models.append(model.to("cuda:0"))

"""### CV Score"""

if 'SOLUTION_14' in ENSEMBLE_SOLUTIONS:

    X_valid = valid[ xgb_feature_cols ]
    y_valid = valid[ CONFIG.target_col ]
    w_valid = valid[ "weight" ]
    y_pred_valid_xgb = xgb_model.predict(X_valid)
    valid_score = r2_score( y_valid, y_pred_valid_xgb, sample_weight=w_valid )
    valid_score

if 'SOLUTION_14' in ENSEMBLE_SOLUTIONS:
    X_valid = valid[ CONFIG.feature_cols ]
    y_valid = valid[ CONFIG.target_col ]
    w_valid = valid[ "weight" ]
    X_valid = X_valid.fillna(method = 'ffill').fillna(0)
    X_valid.shape, y_valid.shape, w_valid.shape

if 'SOLUTION_14' in ENSEMBLE_SOLUTIONS:
    y_pred_valid_nn = np.zeros(y_valid.shape)
    with torch.no_grad():
        for model in models:
            model.eval()
            y_pred_valid_nn += model(torch.FloatTensor(X_valid.values).to("cuda:0")).cpu().numpy() / len(models)
    valid_score = r2_score( y_valid, y_pred_valid_nn, sample_weight=w_valid )
    valid_score

if 'SOLUTION_14' in ENSEMBLE_SOLUTIONS:
    y_pred_valid_ensemble = 0.5 * (y_pred_valid_xgb + y_pred_valid_nn)
    valid_score = r2_score( y_valid, y_pred_valid_ensemble, sample_weight=w_valid )
    valid_score

if 'SOLUTION_14' in ENSEMBLE_SOLUTIONS:
    del valid, X_valid, y_valid, w_valid
    gc.collect()

if 'SOLUTION_14' in ENSEMBLE_SOLUTIONS:

    lags_ : pl.DataFrame | None = None

    def predict_14(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame | pd.DataFrame:
        global lags_
        if lags is not None:
            lags_ = lags

        predictions_14 = test.select(
            'row_id',
            pl.lit(0.0).alias('responder_6'),
        )
        symbol_ids = test.select('symbol_id').to_numpy()[:, 0]

        if not lags is None:
            lags = lags.group_by(["date_id", "symbol_id"], maintain_order=True).last() # pick up last record of previous date
            test = test.join(lags, on=["date_id", "symbol_id"],  how="left")
        else:
            test = test.with_columns(
                ( pl.lit(0.0).alias(f'responder_{idx}_lag_1') for idx in range(9) )
            )

        preds = np.zeros((test.shape[0],))
        preds += xgb_model.predict(test[xgb_feature_cols].to_pandas()) / 2
        test_input = test[CONFIG.feature_cols].to_pandas()
        test_input = test_input.fillna(method = 'ffill').fillna(0)
        test_input = torch.FloatTensor(test_input.values).to("cuda:0")
        with torch.no_grad():
            for i, nn_model in enumerate(tqdm(models)):
                nn_model.eval()
                preds += nn_model(test_input).cpu().numpy() / 10
        print(f"predict> preds.shape =", preds.shape)

        predictions_14 = \
        test.select('row_id').\
        with_columns(
            pl.Series(
                name   = 'responder_6',
                values = np.clip(preds, a_min = -5, a_max = 5),
                dtype  = pl.Float64,
            )
        )

        return predictions_14

"""
=========================
Sub to server
=========================
"""
inference_server = kaggle_evaluation.jane_street_inference_server.JSInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            '/kaggle/input/jane-street-real-time-market-data-forecasting/test.parquet',
            '/kaggle/input/jane-street-real-time-market-data-forecasting/lags.parquet',
        )
    )