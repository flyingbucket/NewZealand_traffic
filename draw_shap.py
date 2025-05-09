import os
import json
import xgboost as xgb
import xgbFunc
import pandas as pd
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(
    description="Draw a shap-value figure based on pretrained xgboost model"
)
parser.add_argument(
    "--one_dir",
    type=bool,
    required=True,
    help="if set true ,this will only draw the shap-value figure of the run you specified with arg --dir",
)
parser.add_argument(
    "--dir", type=str, required=False, help="specify the run you want to draw"
)
parser.add_argument(
    "--model_type",
    type=str,
    required=False,
    help="specify the type of model you want to analyse",
)
args = parser.parse_args()

data_ori = pd.read_csv("data/crash.csv")
base_dir = "result"
# for subdir in os.listdir(base_dir):
subdirs = os.listdir(base_dir)
if args.dir in subdirs:
    subdir = args.dir
else:
    raise ValueError(f"--dir you specified is not under {base_dir}")

subdir_path = os.path.join(base_dir, subdir)
print(f"in dir:{subdir_path}")
if os.path.isdir(subdir_path):
    model_path = os.path.join(subdir_path, "model.json")
    drop_path = os.path.join(subdir_path, "drop.json")
    if os.path.exists(model_path) and os.path.exists(drop_path):
        print("found model.json drop.json")
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
        if args.model_type == "multi_clf":
            figs = xgbFunc.visualize_multiclass_shap(booster, X_train, sample_size=500)
        elif args.model_type == "reg" or args.model_type == "binary_clf":
            figs = xgbFunc.visualize_reg_bin_shap(booster, X_train, sample_size=500)
        else:
            raise ValueError(
                "invalid parse --model_type,must be one of ['multi_clf','reg','binary_clf']"
            )
        for i, fig in enumerate(figs):
            fig.savefig(os.path.join(subdir_path, f"shap_class_{i}.png"), dpi=300)

        print("saved shap fig")
    else:
        print("[x] model.json or drop.json not found")
