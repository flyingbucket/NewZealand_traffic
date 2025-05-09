import os
import shap
import json
import xgboost as xgb
import xgbFunc
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 导出重要性排序得分
# 设置 result 目录路径
base_dir = "result"

# 遍历 base_dir 下所有的子目录
for subdir in os.listdir(base_dir):
    subdir_path = os.path.join(base_dir, subdir)

    # 确保它是一个目录
    if os.path.isdir(subdir_path):
        model_path = os.path.join(subdir_path, "model.json")

        # 检查模型文件是否存在
        if os.path.isfile(model_path):
            # 加载模型
            booster = xgb.Booster()
            booster.load_model(model_path)

            # 遍历几种常用的重要性指标
            for importance_type in ["weight", "gain", "cover"]:
                # 获取特征重要性
                score = booster.get_score(importance_type=importance_type)

                # 转为 DataFrame 并排序
                importance_df = pd.DataFrame(
                    list(score.items()), columns=["Feature", "Importance"]
                ).sort_values(by="Importance", ascending=False)

                # 导出为 CSV 文件
                csv_filename = os.path.join(subdir_path, f"{importance_type}.csv")
                importance_df.to_csv(csv_filename, index=False)

                print(f"[✓] 导出成功：{csv_filename}")
        else:
            print(f"[!] 未找到模型文件：{model_path}")


# data_ori = pd.read_csv("data/crash.csv")
# for subdir in os.listdir(base_dir):
#     subdir_path = os.path.join(base_dir, subdir)
#     if os.path.isdir(subdir_path):
#         model_path = os.path.join(subdir_path, "model.json")
#         drop_path = os.path.join(subdir_path, "drop.json")
#         if os.path.exists(model_path) and os.path.exists(drop_path):
#             booster = xgb.Booster()
#             booster.load_model(model_path)
#             with open(drop_path) as f:
#                 drop_lst = json.load(f)

#             y = data_ori["crashSeverity"]
#             data = xgbFunc.encode_features(data_ori.drop(columns=drop_lst))
#             le_y = LabelEncoder()
#             y = le_y.fit_transform(y)
#             X_train, X_test, y_train, y_test = train_test_split(
#                 data, y, test_size=0.2, random_state=42
#             )
#             shap_fig = xgbFunc.visualize_shap(booster, X_train)
#             shap_fig.savefig(os.path.join(subdir_path, "shap.png"), dpi=300)
#             print("saved shap fig")
