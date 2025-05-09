import argparse
import json
import xgbFunc
import pandas as pd
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

# this is the entry of training-----------------
parser = argparse.ArgumentParser(description="XGBoost Training Script with JSON config")
parser.add_argument(
    "--config", type=str, required=True, help="Path to JSON config file"
)
args = parser.parse_args()

# read config file
with open(args.config, "r") as f:
    config = json.load(f)
print(json.dumps(config, indent=4, ensure_ascii=False))

# define name of this round
round = config["round"]
# drop list of this round
drop_lst = config["drop_lst"]
# randomized search n_trials
n_trials = config["n_trials"]


# result path
result_path = f"result/{round}"
if not os.path.exists(result_path):
    os.mkdir(result_path)
with open(os.path.join(result_path, "drop_lst.json"), "w") as f:
    json.dump(drop_lst, f, indent=4)

# prepare data
data_ori = pd.read_csv(config["data_path"])
y = data_ori[config["target"]]
data = xgbFunc.encode_features(data_ori.drop(columns=drop_lst))
le_y = LabelEncoder()
y = le_y.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    data, y, test_size=0.2, random_state=42
)


# load or search for best_params
param_path = os.path.join(result_path, "param_lst.json")
if os.path.exists(param_path):
    with open(param_path, "r") as f:
        param_lst = json.load(f)
    print("use preloaded params")
else:
    # search for best paramters
    best_params, best_score, label_encoder = xgbFunc.xgb_binary_search_optuna(
        X_train, y_train, n_trials=n_trials, result_path=result_path
    )
    param_lst = [best_params, best_score]
    with open(param_path, "w") as f:
        json.dump(param_lst, f, indent=4)

best_params = param_lst[0]
best_score = param_lst[1]
sample_weights = compute_sample_weight("balanced", y_train)
# define model
model = XGBClassifier(
    objective="binary:logistic",
    n_jobs=-1,
    eval_metric="mlogloss",
    random_state=42,
    tree_method="gpu_hist",
)

# train model
model.set_params(**best_params)
model.fit(X_train, y_train, sample_weight=sample_weights)
y_pred = model.predict(X_test)

importance_figs = xgbFunc.visualize_importance(model)
for fig, imp_type in importance_figs:
    fig.savefig(os.path.join(result_path, f"{imp_type}.png"), dpi=300)
    print(f"saved iportance fig {imp_type}")
model.save_model(os.path.join(result_path, "model.json"))
print("saved model!")
