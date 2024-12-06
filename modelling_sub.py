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