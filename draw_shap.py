import os
import json
import xgboost as xgb
import xgbFunc
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data_ori = pd.read_csv("data/crash.csv")
base_dir = "result"
for subdir in os.listdir(base_dir):
    subdir_path = os.path.join(base_dir, subdir)
    print(f"in dir:{subdir_path}")
    if os.path.isdir(subdir_path):
        model_path = os.path.join(subdir_path, "model.json")
        drop_path = os.path.join(subdir_path, "drop.json")
        if os.path.exists(model_path) and os.path.exists(drop_path):
            print("fount drop.json and drop.json")
            booster = xgb.Booster()
            booster.load_model(model_path)
            with open(drop_path) as f:
                drop_lst = json.load(f)

            y = data_ori["crashSeverity"]
            data = xgbFunc.encode_features(data_ori.drop(columns=drop_lst))
            le_y = LabelEncoder()
            y = le_y.fit_transform(y)
            X_train, X_test, y_train, y_test = train_test_split(
                data, y, test_size=0.2, random_state=42
            )
            shap_fig = xgbFunc.visualize_shap(booster, X_train, sample_size=3000)
            shap_fig.savefig(os.path.join(subdir_path, "shap.png"), dpi=300)
            print("saved shap fig")
        else:
            print("[x] model.json or drop.json not found")
