import os
import numpy as np
import pandas as pd
import lightgbm as lgb

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    return np.mean(np.where(denom == 0, 0, diff / denom)) * 100

print("1. Loading processed data from data/processed/...")
train_df = pd.read_parquet("data/processed/train_tabular.parquet")

# 10 Tabular features
drop_cols = ["sample_id", "fold", "price"]
tab_cols = [c for c in train_df.columns if c not in drop_cols]
X_tab = train_df[tab_cols].values

# 384 Text features
X_text = np.load("data/processed/train_text_emb.npy")

# 512 Vision features
X_img = np.load("data/processed/train_img_emb.npy")

# Stack all 906 features
X = np.hstack([X_tab, X_text, X_img])
print(f"Dataset matrix aligned: {X.shape} (Expected: N rows, 906 columns)")

y = train_df["price"].values
y_log = np.log1p(y)
folds = train_df["fold"].values

oof_preds = np.zeros(len(train_df))

lgb_params = {
    "objective": "regression_l1",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1
}

print("\n2. Training 5-Fold Multimodal LightGBM...")
final_booster = None

for fold in range(5):
    trn_idx = np.where(folds != fold)[0]
    val_idx = np.where(folds == fold)[0]

    trn_data = lgb.Dataset(X[trn_idx], label=y_log[trn_idx])
    val_data = lgb.Dataset(X[val_idx], label=y_log[val_idx], reference=trn_data)

    model = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=1200,
        valid_sets=[trn_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=40, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )

    val_pred = np.clip(np.expm1(model.predict(X[val_idx])), a_min=0.1, a_max=None)
    oof_preds[val_idx] = val_pred
    print(f"Fold {fold} SMAPE: {smape(y[val_idx], val_pred):.2f}%")
    
    final_booster = model

print("-" * 40)
print(f"OVERALL MULTIMODAL OOF SMAPE: {smape(y, oof_preds):.2f}%")
print("-" * 40)

# Save the trained 906-feature booster to models/
os.makedirs("models", exist_ok=True)
final_booster.save_model("models/pricing_lgbm.txt")
print("Saved 906-feature model to models/pricing_lgbm.txt")