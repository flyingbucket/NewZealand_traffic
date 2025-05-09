import os
import json
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ProcessPoolExecutor
from xgbFunc import encode_features, visualize_shap  # 确保这些能被进程池导入

# 并行计算shap value
data_ori = pd.read_csv("data/crash.csv")
base_dir = "result"


def process_subdir(subdir):
    subdir_path = os.path.join(base_dir, subdir)
    print(f"in dir: {subdir_path}")
    if not os.path.isdir(subdir_path):
        return f"[skip] {subdir} is not a directory"

    model_path = os.path.join(subdir_path, "model.json")
    drop_path = os.path.join(subdir_path, "drop.json")
    if not (os.path.exists(model_path) and os.path.exists(drop_path)):
        return f"[x] model.json or drop.json not found in {subdir}"

    try:
        booster = xgb.Booster()
        booster.load_model(model_path)
        with open(drop_path) as f:
            drop_lst = json.load(f)

        y = data_ori["crashSeverity"]
        data = encode_features(data_ori.drop(columns=drop_lst))

        le_y = LabelEncoder()
        y = le_y.fit_transform(y)
        X_train, _, y_train, _ = train_test_split(
            data, y, test_size=0.2, random_state=42
        )

        shap_fig = visualize_shap(booster, X_train, sample_size=3000)
        shap_fig.savefig(os.path.join(subdir_path, "shap.png"), dpi=300)
        return f"[✓] saved shap fig for {subdir}"
    except Exception as e:
        return f"[!] error in {subdir}: {e}"


if __name__ == "__main__":
    subdirs = os.listdir(base_dir)
    with ProcessPoolExecutor(
        max_workers=4
    ) as executor:  # 可根据CPU核心数调整 max_workers
        results = list(executor.map(process_subdir, subdirs))
        for r in results:
            print(r)
