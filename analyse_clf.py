import os
import json
import argparse
import xgboost as xgb
import xgbFunc
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 参数解析
parser = argparse.ArgumentParser(description="Analyse a specified Run of xgboost")
parser.add_argument(
    "--dir", type=str, required=True, help="specify the Run you want to analyse"
)
parser.add_argument(
    "--data", type=str, required=True, help="path to data you used for training"
)
parser.add_argument(
    "--task",
    choices=["binary", "multi"],
    default="multi",
    help="task type: binary or multi-class",
)
parser.add_argument(
    "--label", type=str, required=True, help="specify he label of this Run"
)
args = parser.parse_args()

# 数据准备
base_dir = "result"
subdir = args.dir
if subdir not in os.listdir(base_dir):
    raise ValueError(f"--dir '{subdir}' not found in '{base_dir}'")
data_ori = pd.read_csv(args.data)

# 分析指定子目录
subdir_path = os.path.join(base_dir, subdir)
print(f"In subdir {subdir}")
if os.path.isdir(subdir_path):
    model_path = os.path.join(subdir_path, "model.json")
    drop_path = os.path.join(subdir_path, "drop.json")
    if os.path.isfile(model_path) and os.path.isfile(drop_path):
        booster = xgb.Booster()
        booster.load_model(model_path)
        with open(drop_path) as f:
            drop_lst = json.load(f)

        y = data_ori[args.label]
        if args.task == "multi":
            data = xgbFunc.encode_features(data_ori.drop(columns=drop_lst))
            le_y = LabelEncoder()
            y = le_y.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(
                data, y, test_size=0.2, random_state=42
            )
            X_test = xgb.DMatrix(X_test)

            y_pred_prob = booster.predict(X_test)

            y_pred = np.argmax(y_pred_prob, axis=1)
        else:  # binary
            data = xgbFunc.encode_features(data_ori.drop(columns=drop_lst))
            le_y = LabelEncoder()
            y = le_y.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(
                data, y, test_size=0.2, random_state=42
            )
            X_test = xgb.DMatrix(X_test)

            y_pred_prob = booster.predict(X_test)
            y_pred = (y_pred_prob > 0.5).astype(int)

        eval_res = xgbFunc.evaluate_multiclass(y_test, y_pred, le_y)
        with open(os.path.join(subdir_path, "evaluation.json"), "w") as f:
            json.dump(eval_res, f, indent=4)

        for importance_type in ["weight", "gain", "cover"]:
            score = booster.get_score(importance_type=importance_type)
            importance_df = pd.DataFrame(
                list(score.items()), columns=["Feature", "Importance"]
            ).sort_values(by="Importance", ascending=False)
            csv_filename = os.path.join(subdir_path, f"{importance_type}.csv")
            importance_df.to_csv(csv_filename, index=False)
            print(f"[✓] 导出成功：{csv_filename}")
    else:
        print(f"[!] 未找到模型文件：{model_path}")
