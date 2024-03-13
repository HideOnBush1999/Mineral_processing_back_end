# coding: utf-8
import platypus as pl
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import time
import numpy as np
from ZOA import ZOA
from SWO import SWO
from KCCOM import KCCMO
import warnings
warnings.filterwarnings('ignore', category=UserWarning,
                        message="X does not have valid feature names, but.*")


plt.rcParams['font.sans-serif'] = ['SimHei']   # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号 #有中文出现的情况，需要u'内容'

file_path = './coal.xlsx'
data = pd.read_excel(file_path)

data = data.drop(data.index[:50])

# 准备数据集
X = data[['入口一次风流量', '入口一次风温度', '给煤量', '磨煤机电流', '原煤温度', '原煤水分']]
y = data['出口煤粉流量']

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 创建并训练SVR模型
svr_model = SVR()
svr_model.fit(X_train, y_train)

# 进行预测
y_pred = svr_model.predict(X_test)

# 计算性能指标
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 输出性能指标
print(f"出口煤粉流量 Mean Squared Error (MSE): {mse}")
print(f"出口煤粉流量 R^2 Score: {r2}")

# 参数的最小值
min_bounds = [27.028415, 121.184768, 21.4084, 67.856317, 10.56, 6.00]
# 参数的最大值
max_bounds = [28.882128, 139.525523, 39.7840, 109.205099, 12.50, 8.87]

# 定义双目标函数


def multi_objective_function(X):
    # 将 X 转换为 DataFrame
    X_df = pd.DataFrame([X], columns=['入口一次风流量', '入口一次风温度',
                        '给煤量', '磨煤机电流', '原煤温度', '原煤水分'])

    # 进行预测
    y_pred = svr_model.predict(X_df)[0]

    # 提取给煤量和磨煤机电流
    x3 = X[2]  # 给煤量
    x4 = X[3]  # 磨煤机电流

    # 第一个目标：最大化 y_pred / x3, 转化为最小化 -y_pred / x3
    objective1 = -(y_pred / x3)

    # 第二个目标：最小化 (x4 - min_x4) / (max_x4 - min_x4)
    objective2 = (x4 - min_bounds[3]) / (max_bounds[3] - min_bounds[3])
    # objective2 = x4

    return [objective1, objective2]


# 使用KCCMO算法进行优化
bounds = [(27.028415, 28.882128), (121.184768, 139.525523), (21.4084, 39.7840),
          (67.856317, 109.205099), (10.56, 12.50), (6.00, 8.87)]

# 在你提供的代码中，helper_function实际上没有被用到。这个函数是为了演示KCCMO算法的通用性而包含的，它通常用于辅助目标函数或约束条件的计算。在这个特定的例子中，由于我们只关注multi_objective_function这个双目标函数，所以helper_function被设置为一个返回常数值的简单函数lambda x: 0。这样做是为了满足KCCMO函数对辅助函数的参数要求，但实际上这个辅助函数在优化过程中没有起到作用。

# 如果你在实际应用中有需要考虑的辅助目标或约束条件，你可以将helper_function替换为相应的函数来计算这些值。例如，如果你需要考虑某些参数的和不超过某个值，你可以定义一个相应的helper_function来计算这个和，并在KCCMO算法中使用它。
start_time = time.time()
best_solution, best_objectives = KCCMO(
    multi_objective_function, lambda x: 0, bounds, num_iterations=100, population_size=50)

end_time = time.time()
print("KCCMO求解运行时间:", end_time - start_time, "秒")

print("最优解:", best_solution)
print("目标函数值:", best_objectives, "\n\n")


# ------------------------------------
# ------------------------------------
# ------------------------------------
X_valve = data[['热风阀门开度', '冷风阀门开度', '热一次风温度', '冷一次风温度']]
y_flow_temperature = data[['入口一次风流量', '入口一次风温度']]

# 分割数据集
X_train_valve, X_test_valve, y_train_flow, y_test_flow = train_test_split(
    X_valve, y_flow_temperature, test_size=0.2, random_state=42)

# 创建多目标XGBoost模型
multi_output_xgb = MultiOutputRegressor(XGBRegressor(
    objective='reg:squarederror', random_state=42))

# 训练模型
multi_output_xgb.fit(X_train_valve, y_train_flow)

# 进行预测
y_pred = multi_output_xgb.predict(X_test_valve)

# 计算性能指标
mse = mean_squared_error(y_test_flow, y_pred)
r2 = r2_score(y_test_flow, y_pred)

# 输出性能指标
print(f"入口一次风流量与温度 Mean Squared Error (MSE): {mse}")
print(f"入口一次风流量与温度 R^2 Score: {r2}")

# 获取最佳入口风流量与温度
best_flow = best_solution[0]
best_temperature = best_solution[1]


def objective_function_valve(X, multi_output_xgb, best_flow, best_temperature):

    # 使用多目标XGBoost模型预测基于热风阀门和冷风阀门开度的入口风流量和温度
    predicted = multi_output_xgb.predict(X.reshape(1, -1))
    # 计算与最佳入口风流量和温度的差异的平方和
    return np.sum((predicted - [best_flow, best_temperature])**2, axis=1)


# 设置热风阀门和冷风阀门开度的上下限
min_valve_bounds = data[['热风阀门开度', '冷风阀门开度', '热一次风温度', '冷一次风温度']].min().values
max_valve_bounds = data[['热风阀门开度', '冷风阀门开度', '热一次风温度', '冷一次风温度']].max().values

# 定义ZOA算法的参数
N = 50   # 群体大小
T = 50  # 最大迭代次数
dim = 4   # 问题维度

start_time = time.time()

# 使用ZOA算法寻找最优解
Best_pos, Best_score, Convergence_curve = ZOA(lambda x: objective_function_valve(x, multi_output_xgb, best_flow, best_temperature),
                                              min_valve_bounds, max_valve_bounds, dim, N, T)

end_time = time.time()
print("ZOA求解运行时间:", end_time - start_time, "秒")

# 输出最优解
print("最优热冷风阀开度和温度:", Best_pos)
print("目标函数的最小值:", Best_score[0], "\n\n")

# 绘制收敛曲线（此处需要根据ZOA的输出进行调整）
plt.plot(Convergence_curve)
plt.xlabel('迭代次数')
plt.ylabel('目标函数值')
plt.title('热冷风阀开度和温度优化目标函数随迭代次数的变化')
plt.show()


# ------------------------------------
# ------------------------------------
# ------------------------------------
# 提取用于模型的特征和目标
X_belt = data[['皮带转速', '比例系数']]
y_coal = data['给煤量']

# 分割数据集
X_train_belt, X_test_belt, y_train_coal, y_test_coal = train_test_split(
    X_belt, y_coal, test_size=0.2, random_state=42)

# 训练SVR模型
svr_model_coal = SVR(kernel='linear')
svr_model_coal.fit(X_train_belt, y_train_coal)

# 进行预测
y_pred = svr_model_coal.predict(X_test_belt)

# 计算性能指标
mse = mean_squared_error(y_test_coal, y_pred)
r2 = r2_score(y_test_coal, y_pred)

# 输出性能指标
print(f"给煤量 Mean Squared Error (MSE): {mse}")
print(f"给煤量 R^2 Score: {r2}")

# 获取最佳给煤量
best_coal = best_solution[2]  # 假设最佳给煤量存储在 pos 数组的第二个位置


def objective_function_belt(X, svr_model, best_coal):
    # 使用SVR模型预测基于皮带转速和比例系数的给煤量
    predicted_coal = svr_model.predict(X.reshape(1, -1))
    # 计算与最佳给煤量的差异
    return np.abs(predicted_coal - best_coal)


# 设置皮带转速和比例系数的上下限
min_belt_bounds = data[['皮带转速', '比例系数']].min().values
max_belt_bounds = data[['皮带转速', '比例系数']].max().values

# 定义SWO算法的参数
SearchAgents_no = 50  # 粒子数
Tmax = 1000            # 最大迭代次数
dim = 2               # 问题维度

start_time = time.time()

# 使用SWO算法寻找最优解
Best_score_belt, Best_SW_belt, Convergence_curve_belt = SWO(
    SearchAgents_no, Tmax, max_belt_bounds, min_belt_bounds, dim, lambda x: objective_function_belt(x, svr_model_coal, best_coal))

end_time = time.time()
print("SWO求解运行时间:", end_time - start_time, "秒")

# 输出最优解
print("最优皮带转速和比例系数:", Best_SW_belt)
print("目标函数的最小值:", Best_score_belt)

# 绘制收敛曲线
plt.plot(Convergence_curve_belt)
plt.xlabel('迭代次数')
plt.ylabel('目标函数值')
plt.title('皮带转速和比例系数优化目标函数随迭代次数的变化')
plt.show()
