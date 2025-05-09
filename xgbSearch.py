from ray import tune
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def train_xgb(config):
    # 数据划分
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    # 参数组合
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "gpu_hist",  # GPU 模式
        "max_depth": config["max_depth"],
        "eta": config["eta"],
        "subsample": config["subsample"],
        "colsample_bytree": config["colsample_bytree"],
        "lambda": config["lambda"]
    }

    # 训练
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=20,
        verbose_eval=False
    )

    preds = bst.predict(dvalid)
    pred_labels = [1 if p > 0.5 else 0 for p in preds]
    acc = accuracy_score(y_valid, pred_labels)
    tune.report(mean_accuracy=acc)

# 搜索空间
search_space = {
    "max_depth": tune.randint(3, 10),
    "eta": tune.loguniform(1e-3, 0.3),
    "subsample": tune.uniform(0.5, 1.0),
    "colsample_bytree": tune.uniform(0.5, 1.0),
    "lambda": tune.loguniform(1e-3, 10.0)
}

# 启动搜索
tuner = tune.Tuner(
    train_xgb,
    param_space=search_space,
    tune_config=tune.TuneConfig(
        num_samples=50,
        metric="mean_accuracy",
        mode="max"
    )
)

results = tuner.fit()
best_result = results.get_best_result(metric="mean_accuracy", mode="max")
print("Best config:", best_result.config)
