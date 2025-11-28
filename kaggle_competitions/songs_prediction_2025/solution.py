import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from lightgbm import early_stopping, log_evaluation

# 1. Загрузка данных
train_x = pd.read_csv("train_x.csv")
train_y = pd.read_csv("train_y.csv")
test_x = pd.read_csv("test_x.csv")

# 2. Выделение и согласование признаков
id_like_cols = ["id", "index", "Unnamed: 0"]

train_features = train_x.copy()
for c in id_like_cols:
    if c in train_features.columns:
        train_features = train_features.drop(columns=[c])

y_train = train_y["year"].astype(float).values

ids = test_x["id"] if "id" in test_x.columns else test_x.iloc[:, 0]
test_features = test_x.copy()
for c in id_like_cols:
    if c in test_features.columns:
        test_features = test_features.drop(columns=[c])

# Согласуем порядок и набор колонок между train и test
train_cols = [c for c in train_features.columns if c in test_features.columns]
X_train = train_features[train_cols].copy()
X_test = test_features.reindex(columns=train_cols, fill_value=0.0).copy()

# 3. KFold CV + LightGBM и Ridge, затем блендинг
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X_train), dtype=float)
oof_ridge = np.zeros(len(X_train), dtype=float)
test_preds_lgb = np.zeros(len(X_test), dtype=float)
test_preds_ridge = np.zeros(len(X_test), dtype=float)

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train), start=1):
    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]

    # LightGBM
    lgbm = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.2,
        random_state=42,
        n_jobs=-1
    )
    lgbm.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="mae",
        callbacks=[
            early_stopping(stopping_rounds=100),
            log_evaluation(period=0)
        ]
    )
    best_iter = getattr(lgbm, "best_iteration_", None)
    oof_lgb[va_idx] = lgbm.predict(X_va, num_iteration=best_iter)
    test_preds_lgb += lgbm.predict(X_test, num_iteration=best_iter) / n_splits

    # Ridge (со стандартизацией)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_test_s = scaler.transform(X_test)

    ridge = Ridge(alpha=2.0)
    ridge.fit(X_tr_s, y_tr)
    oof_ridge[va_idx] = ridge.predict(X_va_s)
    test_preds_ridge += ridge.predict(X_test_s) / n_splits

# 4. Подбор веса блендинга по OOF (минимум MAE)
weights = np.linspace(0.0, 1.0, 51)
maes = []
for w in weights:
    blend = w * oof_lgb + (1.0 - w) * oof_ridge
    maes.append(mean_absolute_error(y_train, blend))
best_w = float(weights[int(np.argmin(maes))])

mae_lgb = mean_absolute_error(y_train, oof_lgb)
mae_ridge = mean_absolute_error(y_train, oof_ridge)
mae_blend = min(maes)

# 5. Прогноз на тесте, постобработка и сохранение
preds_test = best_w * test_preds_lgb + (1.0 - best_w) * test_preds_ridge
pred_years = np.rint(preds_test).astype(int)
pred_years = np.clip(pred_years, 1922, 2011)

submission = pd.DataFrame({
    "id": ids,
    "year": pred_years
})
submission.to_csv("submission.csv", index=False)

print(f"CV MAE LGBM: {mae_lgb:.4f}")
print(f"CV MAE Ridge: {mae_ridge:.4f}")
print(f"CV MAE Blend: {mae_blend:.4f} (best_w={best_w:.2f})")
print("Файл submission.csv создан:")
print(submission.head())
