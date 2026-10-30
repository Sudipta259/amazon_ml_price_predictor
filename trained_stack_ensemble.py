import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import TruncatedSVD

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    return np.mean(np.where(denom == 0, 0, diff / denom)) * 100

print("1. Loading tabular data and embeddings...")
train_df = pd.read_parquet("data/processed/train_tabular.parquet")
test_df = pd.read_parquet("data/processed/test_tabular.parquet")

y = train_df["price"].values
y_log = np.log1p(y)
folds = train_df["fold"].values

train_text = np.load("data/processed/train_text_emb.npy")
test_text = np.load("data/processed/test_text_emb.npy")
train_img = np.load("data/processed/train_img_emb.npy")
test_img = np.load("data/processed/test_img_emb.npy")

# Combine embeddings
X_emb_train = np.hstack([train_text, train_img])
X_emb_test = np.hstack([test_text, test_img])

print("2. Stacking embeddings with Out-of-Fold Ridge Regression...")
ridge_oof = np.zeros(len(train_df))
ridge_test = np.zeros(len(test_df))

for fold in range(5):
    trn_idx = np.where(folds != fold)[0]
    val_idx = np.where(folds == fold)[0]

    ridge = RidgeCV(alphas=np.logspace(-2, 3, 10))
    ridge.fit(X_emb_train[trn_idx], y_log[trn_idx])

    ridge_oof[val_idx] = ridge.predict(X_emb_train[val_idx])
    ridge_test += ridge.predict(X_emb_test) / 5.0

print(f"Ridge OOF SMAPE: {smape(y, np.expm1(ridge_oof)):.2f}%")

print("3. Reducing embedding dimensionality with SVD...")
svd = TruncatedSVD(n_components=16, random_state=42)
svd_train = svd.fit_transform(X_emb_train)
svd_test = svd.transform(X_emb_test)

# Build lean feature matrix
drop_cols = ["sample_id", "fold", "price"]
tab_cols = [c for c in train_df.columns if c not in drop_cols]
X_tab_train = train_df[tab_cols].values
X_tab_test = test_df[tab_cols].values

X = np.hstack([X_tab_train, ridge_oof[:, None], svd_train])
X_test = np.hstack([X_tab_test, ridge_test[:, None], svd_test])
print(f"Compact feature matrix shape: {X.shape}")

print("\n4. Training LightGBM + CatBoost Ensemble...")
lgb_oof = np.zeros(len(train_df))
lgb_test = np.zeros(len(test_df))
cb_oof = np.zeros(len(train_df))
cb_test = np.zeros(len(test_df))

lgb_params = {
    "objective": "regression_l1",
    "metric": "mae",
    "learning_rate": 0.04,
    "num_leaves": 47,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1
}

for fold in range(5):
    trn_idx = np.where(folds != fold)[0]
    val_idx = np.where(folds == fold)[0]

    X_train, y_train = X[trn_idx], y_log[trn_idx]
    X_val, y_val = X[val_idx], y_log[val_idx]

    # LightGBM
    trn_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=trn_data)
    lgb_model = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=1200,
        valid_sets=[trn_data, val_data],
        callbacks=[lgb.early_stopping(40, verbose=False)]
    )
    lgb_oof[val_idx] = np.expm1(lgb_model.predict(X_val))
    lgb_test += np.expm1(lgb_model.predict(X_test)) / 5.0

    # CatBoost
    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=42,
        verbose=0
    )
    cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=40)
    cb_oof[val_idx] = np.expm1(cb_model.predict(X_val))
    cb_test += np.expm1(cb_model.predict(X_test)) / 5.0

# 5. Blend Predictions (60% LightGBM + 40% CatBoost)
final_oof = np.clip(0.6 * lgb_oof + 0.4 * cb_oof, a_min=0.1, a_max=None)
final_test = np.clip(0.6 * lgb_test + 0.4 * cb_test, a_min=0.1, a_max=None)

print("\n" + "=" * 45)
print(f"LightGBM OOF SMAPE:  {smape(y, lgb_oof):.2f}%")
print(f"CatBoost OOF SMAPE:  {smape(y, cb_oof):.2f}%")
print(f"ENSEMBLE OOF SMAPE:  {smape(y, final_oof):.2f}%")
print("=" * 45)

os.makedirs("submissions", exist_ok=True)
pd.DataFrame({"sample_id": test_df["sample_id"], "price": final_test}).to_csv(
    "submissions/submission_ensemble.csv", index=False
)
print("Saved final ensemble predictions to submissions/submission_ensemble.csv")