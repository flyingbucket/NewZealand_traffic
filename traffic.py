# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: xgb-gpu
#     language: python
#     name: python3
# ---

import xgbFunc
import xgboost as xgb
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import shap
import importlib
importlib.reload(xgbFunc)

# from xgboost import XGBClassifier
# clf = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor')
# print(clf)
# from xgboost import Booster
# print(Booster({'tree_method': 'gpu_hist'}).attributes())


# +
# import numpy as np
# from xgboost import XGBClassifier

# X = np.random.rand(100, 10)
# y = np.random.randint(0, 2, size=100)

# model = XGBClassifier(tree_method='hist', device='gpu')
# model.fit(X, y)

# -

data_ori=pd.read_csv("data/crash.csv")
data_ori.head()

y = data_ori["crashSeverity"]
data=xgbFunc.encode_features(data_ori.drop(columns=["crashSeverity"]))
data.head()

data.columns

le_y=LabelEncoder()
y=le_y.fit_transform(y)
X_train, X_test, y_train, y_test=train_test_split(data,y, test_size=0.2, random_state=42)


best_params,best_score,label_encoder,num_classes=xgbFunc.xgb_clf_search_gpu(X_train,y_train,n_iter=100)

model = XGBClassifier(
    objective='multi:softprob',
    num_class=num_classes,
    n_jobs=-1,
    eval_metric='mlogloss',
    random_state=42,
    tree_method='hist',     
    device='cuda',         
)
model.set_params(**best_params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


import os
if not os.path.exists("result"):
    os.makedirs("result")
result_path="result"
importance_figs=xgbFunc.visualize_importance(model)
for fig,imp_type in importance_figs:
    fig.savefig(os.path.join(result_path,f"{imp_type}.png"), dpi=300)

model.save_model(os.path.join(result_path,"model_ori.json"))

# +
# shap_fig=xgbFunc.visulize_shap(model, X_train)
# shap_fig.savefig(os.path.join(result_path,"shap.png"), dpi=300)
