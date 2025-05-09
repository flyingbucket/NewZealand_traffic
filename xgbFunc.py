import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from scipy.stats import uniform, randint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def encode_features(df: pd.DataFrame, fill_value="__missing__"):
    df = df.copy()  # 避免修改原始数据
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not categorical_cols:
        print("没有非数值类型的列，无需编码。")
        return df

    # 填充缺失值
    df[categorical_cols] = df[categorical_cols].fillna(fill_value)

    # 编码器
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    # 拟合并替换原列
    df[categorical_cols] = encoder.fit_transform(df[categorical_cols])

    return df


def xgb_reg_search(X_train, y_train):
    param_dist = {
        "n_estimators": randint(100, 300),  # 树的数量
        "learning_rate": uniform(0.01, 0.3),  # 学习率
        "max_depth": randint(3, 10),  # 树的最大深度
        "min_child_weight": randint(1, 10),  # 子节点所需最小样本数
        "gamma": uniform(0, 0.5),  # 最小损失减少量
        "subsample": uniform(0.7, 0.3),  # 样本采样比例
        "colsample_bytree": uniform(0.7, 0.3),  # 特征采样比例
        "scale_pos_weight": uniform(0.5, 3),  # 类别不平衡的加权因子
        "alpha": uniform(0, 1),  # L1正则化项
        "lambda": uniform(0, 1),  # L2正则化项
    }

    model = XGBRegressor(
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=100,
        scoring="neg_mean_squared_error",
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42,
        return_train_score=True,
    )
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    print("Best Parameters:")
    for key, value in best_params.items():
        print(f"{key}: {value}")
    print("Best Score:", best_score)
    return best_params, best_score


def xgb_clf_search_gpu_0(X_train, y_train, n_iter=100):
    # 自动编码字符串标签
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)
    num_classes = len(np.unique(y_encoded))

    param_dist = {
        "n_estimators": randint(100, 300),
        "learning_rate": uniform(0.01, 0.3),
        "max_depth": randint(3, 10),
        "min_child_weight": randint(1, 10),
        "gamma": uniform(0, 0.5),
        "subsample": uniform(0.7, 0.3),
        "colsample_bytree": uniform(0.7, 0.3),
        "reg_alpha": uniform(0, 1),
        "reg_lambda": uniform(0, 1),
    }

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist",  # 使用 GPU 训练
        device="cuda",
        n_jobs=-1,
    )

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="accuracy",
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42,
        return_train_score=True,
    )

    random_search.fit(X_train, y_encoded)

    best_params = random_search.best_params_
    best_score = random_search.best_score_

    print("Best Parameters:")
    for key, value in best_params.items():
        print(f"{key}: {value}")
    print("Best Accuracy:", best_score)

    return best_params, best_score, label_encoder, num_classes


def xgb_clf_search(X_train, y_train, n_iter=100):
    # 自动编码字符串标签
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)
    num_classes = len(np.unique(y_encoded))

    param_dist = {
        "n_estimators": randint(100, 300),
        "learning_rate": uniform(0.01, 0.3),
        "max_depth": randint(3, 10),
        "min_child_weight": randint(1, 10),
        "gamma": uniform(0, 0.5),
        "subsample": uniform(0.7, 0.3),
        "colsample_bytree": uniform(0.7, 0.3),
        "reg_alpha": uniform(0, 1),  # 注意参数名不同
        "reg_lambda": uniform(0, 1),
    }

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_jobs=-1,
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist",
        device="gpu",
    )

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="accuracy",
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42,
        return_train_score=True,
    )

    random_search.fit(X_train, y_encoded)

    best_params = random_search.best_params_
    best_score = random_search.best_score_

    print("Best Parameters:")
    for key, value in best_params.items():
        print(f"{key}: {value}")
    print("Best Accuracy:", best_score)

    return best_params, best_score, label_encoder, num_classes


def evaluate(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R^2:", r2)
    return mse, rmse, mae, r2


def visualize_importance(model):
    figs = []

    # Feature Importance - weight
    fig1, ax1 = plt.subplots()
    fig1, ax1 = plt.subplots(figsize=(15, 15))
    fig1.subplots_adjust(left=0.18)
    xgb.plot_importance(model, importance_type="weight", ax=ax1)
    ax1.set_title("Feature Importance (Weight)")
    figs.append((fig1, "weight"))

    # Feature Importance - gain
    fig2, ax2 = plt.subplots()
    fig2, ax2 = plt.subplots(figsize=(15, 15))
    fig2.subplots_adjust(left=0.18)
    xgb.plot_importance(model, importance_type="gain", ax=ax2)
    ax2.set_title("Feature Importance (Gain)")
    figs.append((fig2, "gain"))

    # Feature Importance - cover
    fig3, ax3 = plt.subplots()
    fig3, ax3 = plt.subplots(figsize=(15, 15))
    fig3.subplots_adjust(left=0.18)
    xgb.plot_importance(model, importance_type="cover", ax=ax3)
    ax3.set_title("Feature Importance (Cover)")
    figs.append((fig3, "cover"))
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)

    return figs


def visualize_shap(model, X_train, sample_size=10000):
    booster = model if isinstance(model, xgb.Booster) else model.get_booster()
    if len(X_train) > sample_size:
        idx = np.random.choice(len(X_train), sample_size, replace=False)
        X_sample = X_train.iloc[idx] if hasattr(X_train, "iloc") else X_train[idx]
    else:
        X_sample = X_train
    explainer = shap.TreeExplainer(booster, X_train)
    shap_values = explainer(X_sample)
    shap.summary_plot(
        shap_values,
        X_train,
        feature_names=X_train.columns.tolist(),
        show=False,
        plot_type="dot",
    )
    fig = plt.gcf()
    fig.set_size_inches(10, 12)
    plt.title("SHAP Summary Plot")
    return fig


def xgb_clf_search_gpu(X, y, n_iter=100, cv=3, random_state=42):
    rng = np.random.RandomState(random_state)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(np.unique(y_encoded))

    # 创建 GPU DMatrix 数据，一次性复制到 GPU
    dmatrix = xgb.DMatrix(X, label=y_encoded)

    # 参数空间
    param_dist = {
        "learning_rate": lambda: uniform(0.01, 0.3).rvs(random_state=rng),
        "max_depth": lambda: randint(3, 10).rvs(random_state=rng),
        "min_child_weight": lambda: randint(1, 10).rvs(random_state=rng),
        "gamma": lambda: uniform(0, 0.5).rvs(random_state=rng),
        "subsample": lambda: uniform(0.7, 0.3).rvs(random_state=rng),
        "colsample_bytree": lambda: uniform(0.7, 0.3).rvs(random_state=rng),
        "reg_alpha": lambda: uniform(0, 1).rvs(random_state=rng),
        "reg_lambda": lambda: uniform(0, 1).rvs(random_state=rng),
        "n_estimators": lambda: randint(100, 300).rvs(random_state=rng),
    }

    # K折交叉验证划分
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    best_score = 0
    best_params = None

    print("Starting random search...")

    for i in tqdm(range(n_iter)):
        # 随机采样一组参数
        sampled_params = {k: sampler() for k, sampler in param_dist.items()}

        # 设置 XGBoost 参数
        xgb_params = {
            "objective": "multi:softprob",
            "num_class": num_classes,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "device": "cuda",
            "learning_rate": sampled_params["learning_rate"],
            "max_depth": int(sampled_params["max_depth"]),
            "min_child_weight": sampled_params["min_child_weight"],
            "gamma": sampled_params["gamma"],
            "subsample": sampled_params["subsample"],
            "colsample_bytree": sampled_params["colsample_bytree"],
            "reg_alpha": sampled_params["reg_alpha"],
            "reg_lambda": sampled_params["reg_lambda"],
        }

        n_estimators = int(sampled_params["n_estimators"])
        scores = []

        # 手动做 K 折交叉验证
        for train_idx, valid_idx in skf.split(X, y_encoded):
            dtrain = dmatrix.slice(train_idx)
            dvalid = dmatrix.slice(valid_idx)

            evals = [(dvalid, "eval")]
            booster = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=n_estimators,
                evals=evals,
                verbose_eval=False,
            )

            preds = booster.predict(dvalid)
            # print(np.unique(preds))
            acc = (preds.argmax(axis=1) == y_encoded[valid_idx]).mean()
            scores.append(acc)

        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params = sampled_params

    print("\nBest Parameters:")
    for k, v in best_params.items():
        print(f"{k}: {v}")
    print("Best Accuracy:", best_score)

    return best_params, best_score, label_encoder, num_classes
