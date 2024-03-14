import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# 内置特征重要性函数
def feature_importance_model(model, X, y, file_name="feature_importance_model.xlsx"):
    feature_importances = model.feature_importances_
    indices = np.argsort(feature_importances)[::-1]

    # 打印特征重要性
    print("Feature ranking (Model based):")
    for f in range(X.shape[1]):
        print(
            f"{f + 1}. feature {X.columns[indices[f]]} ({feature_importances[indices[f]]})")

    # # 选择前十个特征
    # top_features = X.columns[indices[:10]]

    # # 保存到新的 DataFrame
    # df_top_features = pd.concat([y, X[top_features]], axis=1)
    # df_top_features.to_excel(file_name, index=False)


# 排列特征重要性函数
def permutation_feature_importance(model, X, y, file_name="permutation_feature_importance.xlsx"):
    perm_importance = permutation_importance(
        model, X, y, n_repeats=10, random_state=42)
    sorted_idx = perm_importance.importances_mean.argsort()[::-1]

    # 打印特征重要性
    print("Feature ranking (Permutation based):")
    for i in sorted_idx[:10]:
        print(f"{X.columns[i]}: {perm_importance.importances_mean[i]}")

    # # 选择前十个特征
    # top_features = X.columns[sorted_idx[:10]]

    # # 保存到新的 DataFrame
    # df_top_features = pd.concat([y, X[top_features]], axis=1)
    # df_top_features.to_excel(file_name, index=False)


# 递归特征消除函数
def recursive_feature_elimination(model, X, y, file_name="recursive_feature_elimination.xlsx",
                                  n_features_to_select=10):
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
    rfe.fit(X, y)

    # 打印特征排名
    print("Feature ranking (RFE based):")
    for i in range(X.shape[1]):
        if rfe.ranking_[i] <= n_features_to_select:
            print(f"{X.columns[i]}: {rfe.ranking_[i]}")

    # # 选择前十个特征
    # top_features = X.columns[rfe.support_]

    # # 保存到新的 DataFrame
    # df_top_features = pd.concat([y, X[top_features]], axis=1)
    # df_top_features.to_excel(file_name, index=False)


def evaluate_feature_importance(file_path):
    df = pd.read_excel(file_path)

    X = df.iloc[:, 1:]  # 输入特征：第二列到最后一列
    y = df.iloc[:, 0]   # 目标变量：第一列

    # 分割数据为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # 构建随机森林模型
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # 训练模型
    model.fit(X_train, y_train)

    # 预测测试集
    y_pred = model.predict(X_test)

    # 评估模型
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'R² Score: {r2}')


if __name__ == '__main__':
    file_path = "./modified_data.xlsx"
    df = pd.read_excel(file_path)

    X = df.iloc[:, 1:]  # 输入特征：第二列到最后一列
    y = df.iloc[:, 0]   # 目标变量：第一列

    model_path = "model/RandomForest_20231111-180348.joblib"
    model = load(model_path)

    feature_importance_model(model, X, y)
    permutation_feature_importance(model, X, y)
    recursive_feature_elimination(model, X, y)

    # evaluate_feature_importance(file_path="./feature_importance_model.xlsx")
    # evaluate_feature_importance(
    #     file_path="./permutation_feature_importance.xlsx")
    # evaluate_feature_importance(
    #     file_path="./recursive_feature_elimination.xlsx")
