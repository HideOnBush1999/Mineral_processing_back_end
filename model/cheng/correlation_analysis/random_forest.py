from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
import datetime
import os


def get_best_n_estimators(df):
    # 准备数据
    X = df.iloc[:, 1:]  # 输入特征：第二列到最后一列
    y = df.iloc[:, 0]   # 目标变量：第一列

    # 分割数据为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # 定义一个参数网格来进行搜索
    param_grid = {
        'n_estimators': [10, 50, 100, 200, 250, 300, 350]  # 您可以添加更多要搜索的值
    }

    # 创建一个随机森林回归器
    rf_model = RandomForestRegressor(random_state=42)

    # 创建 GridSearchCV
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid,
                               cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

    # 在训练数据上拟合 GridSearchCV
    grid_search.fit(X_train, y_train.ravel())

    # 从网格搜索中获取最佳超参数
    best_n_estimators = grid_search.best_params_['n_estimators']
    print("Best number of estimators (n_estimators):", best_n_estimators)

    return best_n_estimators


def build_and_train_model(df, best_n_estimators):
    # 准备数据
    X = df.iloc[:, 1:]  # 输入特征：第二列到最后一列
    y = df.iloc[:, 0]   # 目标变量：第一列

    # 分割数据为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # 构建随机森林模型
    model = RandomForestRegressor(
        n_estimators=best_n_estimators, random_state=42)

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


    # 绘制真值与预测值
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # 绘制 y=x 对角线
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True Values vs Predictions')
    plt.show()

    # plt.figure(figsize=(10, 6))
    # plt.plot(y_test.reset_index(drop=True)[::50], label='True Values', marker='o', color='blue')  # 每隔 50 个点采样
    # # plt.plot(y_test[::50], label='True Values', marker='o', color='blue')
    # plt.plot(y_pred[::50], label='Predictions', marker='x', color='orange')   # 每隔 50 个点采样
    # plt.xlabel('Data Point Index')
    # plt.ylabel('Target Value')
    # plt.title('True Values vs Predictions')
    # plt.legend()
    # plt.show()


    return model


def save_model(model):
    # 创建 model 文件夹（如果它不存在）
    model_directory = "model"
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    # 获取当前时间
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # 模型文件的完整路径
    model_name = f"RandomForest_{current_time}.joblib"
    model_path = os.path.join(model_directory, model_name)

    # 保存模型
    dump(model, model_path)

    print(f"模型已保存为: {model_path}")


if __name__ == '__main__':
    file_path = "./modified_data.xlsx"
    df = pd.read_excel(file_path)
    # best_n_estimators = get_best_n_estimators(df_normalized)
    best_n_estimators = 350
    model = build_and_train_model(df, best_n_estimators)
    # save_model(model)
