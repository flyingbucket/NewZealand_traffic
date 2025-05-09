import json
import xgbFunc
import pandas as pd
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# entry of training-----------------
# prepare data
data_ori = pd.read_csv("data/crash.csv")
y = data_ori["crashSeverity"]
drop_lst = ["crashSeverity", "minorInjuryCount", "seriousInjuryCount", "fatalCount"]
data = xgbFunc.encode_features(data_ori.drop(columns=drop_lst))
le_y = LabelEncoder()
y = le_y.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    data, y, test_size=0.2, random_state=42
)

round = "no_injury"
result_path = f"result/{round}"
if not os.path.exists(result_path):
    os.mkdir(result_path)

# load or search for best_params
param_path = os.path.join(result_path, "param_lst.json")
if os.path.exists(param_path):
    with open(param_path, "r") as f:
        param_lst = json.load(f)
else:
    # search for best paramters
    best_params, best_score, label_encoder, num_class = xgbFunc.xgb_clf_search_gpu(
        X_train, y_train, n_iter=100
    )
    param_lst = [best_params, best_score, num_class]
    with open(param_path, "w") as f:
        json.dump(param_lst, f, indent=4)

best_params = param_lst[0]
best_score = param_lst[1]
num_class = param_lst[2]

# define model
model = XGBClassifier(
    objective="multi:softprob",
    num_class=num_class,
    n_jobs=-1,
    eval_metric="mlogloss",
    random_state=42,
    tree_method="hist",
    device="cuda",
)

# train model
model.set_params(**best_params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

importance_figs = xgbFunc.visualize_importance(model)
for fig, imp_type in importance_figs:
    fig.savefig(os.path.join(result_path, f"{imp_type}.png"), dpi=300)

model.save_model(os.path.join(result_path, "model.json"))
