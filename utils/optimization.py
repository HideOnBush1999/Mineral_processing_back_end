import numpy as np
import pandas as pd
from pyswarm import pso
from utils.database import get_redis_client
from utils.logger import logger
from utils.database import get_minio_client
import os
import joblib
from scipy.interpolate import interp1d


# PSO 算法在定义的搜索空间内进行如下步骤：

# 初始化粒子的位置和速度。
# 计算每个粒子在当前位置的适应度值（即目标函数值）。
# 更新每个粒子的速度和位置，使其向着当前最佳位置和全局最佳位置移动。
# 反复迭代，直到达到最大迭代次数或者满足收敛条件。


def get_bounds(dataset_name, module_name):
    # 读取数据集
    local_dir = './data/multi_layer'
    dataset_path = os.path.join(local_dir, dataset_name)
    logger.info("读取数据集：{}".format(dataset_path))
    try:
        data = pd.read_excel(dataset_path)
    except Exception as e:
        logger.error("数据集读取失败：{}".format(e))

    if module_name == "给煤机" or module_name == "给风机" or module_name == "磨煤":
        data = data.drop(data.index[:50])
        if module_name == "给煤机":
            input_cols = ['皮带转速', '比例系数']

        if module_name == "给风机":
            input_cols = ['热风阀门开度', '冷风阀门开度', '热一次风温度', '冷一次风温度']

        if module_name == "磨煤":
            input_cols = ['入口一次风流量', '入口一次风温度', '给煤量', '磨煤机电流', '原煤温度', '原煤水分']

    if module_name == "锅炉进口空预器" or module_name == "给水系统" or module_name == "锅炉燃烧":
        data = data.dropna()
        num_samples = 1000
        data = interpolate_data(data, num_samples)

        if module_name == "锅炉进口空预器":
            input_cols = ['O2 in APH (%)', 'Flue Gas in Temperature (°C)',
                          'Flue gas temperature (℃)']

        if module_name == "给水系统":
            input_cols = [
                'Superheater desuperheating water flow (t/h)',
                'Reheater desuperheating water flow (t/h)',
                'Feedwater pressure (MPa)',
                'Flue gas temperature (℃)',
                'Circulating water outlet temperature (℃)'
            ]

        if module_name == "锅炉燃烧":
            input_cols = [
                'Coal Flow (t/h)',
                'O2 Out APH (%)',
                'Corrected Flue Gas Out Temperature (°C)',
                'Feedwater temperature (℃)',
                'Feedwater flow (t/h)',
                'Energy Input From Boiler (Kcal/h)',
                'Boiler oxygen level (%)'
            ]

    input_bounds = []
    for col in input_cols:
        input_bounds.append((data[col].min(), data[col].max()))

    bounds = tuple(input_bounds)
    logger.info("输入参数上下限：{}".format(bounds))
    return bounds


def get_model(model_name):
    # 读取模型
    model_dir = './model/multi-layer-model'
    model_path = os.path.join(model_dir, model_name)

    # 如果本地不存在，则去 MinIO 下载
    if not os.path.exists(model_path):
        minio_client = get_minio_client()
        bucket_name = 'multi-layer-model'

        minio_client.fget_object(
            bucket_name,
            model_name,
            model_path
        )

    # 读取模型
    model = joblib.load(model_path)
    logger.info(f'Loaded model {model_name} from {model_path}')
    return model


def interpolate_data(df, num_points):
    x_old = np.arange(len(df))
    x_new = np.linspace(0, len(df) - 1, num=num_points)
    df_interpolated = pd.DataFrame()

    for col in df.columns:
        f = interp1d(x_old, df[col], kind='quadratic',
                     fill_value="extrapolate")
        df_interpolated[col] = f(x_new)

    return df_interpolated


class DataNotFoundError(Exception):
    pass


# 磨煤的目标函数
def objective_function_grinding(x, dataset_name, model):
    file_path = f'./data/multi_layer/{dataset_name}'
    df = pd.read_excel(file_path)
    df_cleaned = df.drop(df.index[:50])

    # 获取相关列的最大值和最小值
    current_min = df_cleaned['磨煤机电流'].min()
    current_max = df_cleaned['磨煤机电流'].max()

    # x 是一个包含 new_input_columns 中的所有属性值的数组
    new_input_columns = ['入口一次风流量', '入口一次风温度', '给煤量', '磨煤机电流', '原煤温度', '原煤水分']
    X = pd.DataFrame([x], columns=new_input_columns)

    # 预测输出
    y_pred = model.predict(X)
    logger.info(f"y_pred.shape: {y_pred.shape}")
    logger.info(f"y_pred[0]: {y_pred[0]}")

    # 计划目标1：最大化产出比
    object1 = y_pred[0] / x[new_input_columns.index('给煤量')]

    # 计划目标2：耗电量尽量小
    object2 = (x[new_input_columns.index('磨煤机电流')] -
               current_min) / (current_max - current_min)

    combined_objective = -object1 + object2

    return combined_objective


# 给煤机的目标函数
def objective_function_coal_machine(x, model):
    input_columns_coal_machine = ['皮带转速', '比例系数']

    # x 是一个包含 input_columns_coal_machine 中的所有属性值的数组
    X = pd.DataFrame([x], columns=input_columns_coal_machine)

    # 预测输出
    y_pred = model.predict(X)
    logger.info(f"y_pred.shape: {y_pred.shape}")
    logger.info(f"y_pred[0]: {y_pred[0]}")

    # 计算目标：给煤量 与磨煤优化后的最佳输入的差值
    redis_client = get_redis_client()
    coal_supply = redis_client.get('coal_supply')
    if coal_supply is None:
        raise DataNotFoundError('coal_supply not found in Redis')

    coal_supply_diff = abs(y_pred[0] - float(coal_supply))

    return coal_supply_diff

# 给风机的目标函数


def objective_function_wind_machine(x, model):
    input_columns_wind_machine = ['热风阀门开度', '冷风阀门开度', '热一次风温度', '冷一次风温度']

    X = pd.DataFrame([x], columns=input_columns_wind_machine)

    # 预测输出
    y_pred = model.predict(X)
    logger.info(f"y_pred.shape: {y_pred.shape}")
    logger.info(f"y_pred[0]: {y_pred[0]}")

    # 计算目标：入口一次风流量 和 入口一次风温度 与磨煤优化后的最佳输入的差值
    redis_client = get_redis_client()
    air_flow = redis_client.get('air_flow')
    air_temperature = redis_client.get('air_temperature')
    if air_flow is None or air_temperature is None:
        raise DataNotFoundError(
            'air_flow or air_temperature not found in Redis')

    air_flow_diff = abs(y_pred[0][0] - float(air_flow))
    air_temperature_diff = abs(y_pred[0][1] - float(air_temperature))

    combined_objective = air_flow_diff + air_temperature_diff

    return combined_objective


# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

# 锅炉燃烧的目标函数
def objective_function_boiler(x, dataset_name, model):
    file_path = f'./data/multi_layer/{dataset_name}'
    df = pd.read_excel(file_path, engine='openpyxl')
    df_cleaned = df.dropna()

    # 获取输出列的最大值和最小值
    so2_min = df_cleaned['SO2 (mg/m3)'].min()
    so2_max = df_cleaned['SO2 (mg/m3)'].max()
    nox_min = df_cleaned['Nox (mg/m3)'].min()
    nox_max = df_cleaned['Nox (mg/m3)'].max()
    co_min = df_cleaned['CO (mg/m3)'].min()
    co_max = df_cleaned['CO (mg/m3)'].max()
    co2_min = df_cleaned['CO2 (ppm)'].min()
    co2_max = df_cleaned['CO2 (ppm)'].max()

    # x 是一个包含 new_input_columns 中的所有属性值的数组
    new_input_columns = [
        'Coal Flow (t/h)',
        'O2 Out APH (%)',
        'Corrected Flue Gas Out Temperature (°C)',
        'Feedwater temperature (℃)',
        'Feedwater flow (t/h)',
        'Energy Input From Boiler (Kcal/h)',
        'Boiler oxygen level (%)'
    ]
    X = pd.DataFrame([x], columns=new_input_columns)

    # 预测输出
    y_pred = model.predict(X)

    # 计算目标 1: 最大化 Boiler Eff (%)
    boiler_eff = y_pred[0][0]

    # 计算目标 2: 最小化 SO2, Nox, CO, CO2 的和（分别归一化）
    emissions = y_pred[0][1:5]
    emissions_normalized = [
        (emissions[0] - so2_min) / (so2_max - so2_min),
        (emissions[1] - nox_min) / (nox_max - nox_min),
        (emissions[2] - co_min) / (co_max - co_min),
        (emissions[3] - co2_min) / (co2_max - co2_min)
    ]
    emissions_sum_normalized = np.sum(emissions_normalized)

    # 将两个目标函数结合起来作为单一目标函数，使用加权求和法
    # 可以调整权重来平衡两个目标的影响
    weight_efficiency = 0.5
    weight_emissions = 0.5
    combined_objective = -weight_efficiency * boiler_eff + \
        weight_emissions * emissions_sum_normalized

    return combined_objective


# 锅炉空预器的目标函数
def objective_function_aph(x, model):
    input_columns_aph = [
        'O2 in APH (%)', 'Flue Gas in Temperature (°C)', 'Flue gas temperature (℃)']

    # x 是一个包含 input_columns_aph 中的所有属性值的数组
    X = pd.DataFrame([x], columns=input_columns_aph)

    # 预测输出
    y_pred = model.predict(X)

    # 计算目标：O2 Out APH (%) 和 Corrected Flue Gas Out Temperature (°C) 与锅炉燃烧优化后的最佳输入的差值
    redis_client = get_redis_client()
    o2_out_aph = redis_client.get(f"O2_Out_APH")
    corrected_temp = redis_client.get(f"Corrected_Flue_Gas_Out_Temperature")
    if o2_out_aph is None or corrected_temp is None:
        raise DataNotFoundError(
            "O2_Out_APH or Corrected_Flue_Gas_Out_Temperature is None")

    o2_out_aph_diff = abs(y_pred[0][0] - float(o2_out_aph))
    corrected_temp_diff = abs(y_pred[0][1] - float(corrected_temp))

    # 目标函数为两个差值的绝对值之和
    combined_objective = o2_out_aph_diff + corrected_temp_diff

    return combined_objective


# 给水系统的目标函数
def objective_function_water(x, model):
    input_columns_water = [
        'Superheater desuperheating water flow (t/h)',
        'Reheater desuperheating water flow (t/h)',
        'Feedwater pressure (MPa)',
        'Flue gas temperature (℃)',
        'Circulating water outlet temperature (℃)'
    ]
    # x 是一个包含 input_columns_water 中的所有属性值的数组
    X = pd.DataFrame([x], columns=input_columns_water)

    # 预测输出
    y_pred = model.predict(X)

    # 计算目标：Feedwater temperature (℃) 和 Feedwater flow (t/h) 与锅炉燃烧优化后的最佳输入的差值
    redis_client = get_redis_client()
    feedwater_temp = redis_client.get(f"Feedwater_Temperature")
    feedwater_flow = redis_client.get(f"Feedwater_Flow")
    if feedwater_temp is None or feedwater_flow is None:
        raise DataNotFoundError(
            "Feedwater_Temperature or Feedwater_Flow is None")

    feedwater_temp_diff = abs(y_pred[0][0] - float(feedwater_temp))
    feedwater_flow_diff = abs(y_pred[0][1] - float(feedwater_flow))

    # 目标函数为两个差值的绝对值之和
    combined_objective = feedwater_temp_diff + feedwater_flow_diff

    return combined_objective


def get_optimization_results(dataset_name, module_name, model, bounds, particles, iterations):
    if module_name == "磨煤":
        def objective_function(x): return objective_function_grinding(
            x, dataset_name=dataset_name, model=model)
        optimal_inputs, optimal_value = pso(objective_function, lb=[b[0] for b in bounds], ub=[
                                            b[1] for b in bounds], swarmsize=particles, maxiter=iterations)

        # 将需要的中间结果写入到 redis 数据库中
        redis_client = get_redis_client()
        redis_client.set(f"air_flow", optimal_inputs[0])
        redis_client.set(f"air_temperature", optimal_inputs[1])
        redis_client.set(f"coal_supply", optimal_inputs[2])

        input_keys = ['入口一次风流量', '入口一次风温度', '给煤量', '磨煤机电流', '原煤温度', '原煤水分']
        optimal_inputs = dict(zip(input_keys, optimal_inputs))

        return optimal_inputs, optimal_value

    if module_name == "给煤机":
        def objective_function(x): return objective_function_coal_machine(
            x, model=model)
        optimal_inputs, optimal_value = pso(objective_function, lb=[b[0] for b in bounds], ub=[
                                            b[1] for b in bounds], swarmsize=particles, maxiter=iterations)

        input_keys = ['皮带转速', '比例系数']
        optimal_inputs = dict(zip(input_keys, optimal_inputs))
        print("optimal_inputs: ", optimal_inputs)

        return optimal_inputs, optimal_value

    if module_name == "给风机":
        def objective_function(x): return objective_function_wind_machine(
            x, model=model)
        optimal_inputs, optimal_value = pso(objective_function, lb=[b[0] for b in bounds], ub=[
                                            b[1] for b in bounds], swarmsize=particles, maxiter=iterations)

        input_keys = ['热风阀门开度', '冷风阀门开度', '热一次风温度', '冷一次风温度']
        optimal_inputs = dict(zip(input_keys, optimal_inputs))

        return optimal_inputs, optimal_value

    if module_name == "锅炉燃烧":
        def objective_function(x): return objective_function_boiler(
            x, dataset_name=dataset_name, model=model)
        optimal_inputs, optimal_value = pso(objective_function, lb=[b[0] for b in bounds], ub=[
                                            b[1] for b in bounds], swarmsize=particles, maxiter=iterations)

        # 将需要的中间结果写入到 redis 数据库中
        redis_client = get_redis_client()
        redis_client.set(f"O2_Out_APH", optimal_inputs[1])
        redis_client.set(
            f"Corrected_Flue_Gas_Out_Temperature", optimal_inputs[2])
        redis_client.set(f"Feedwater_Temperature", optimal_inputs[3])
        redis_client.set(f"Feedwater_Flow", optimal_inputs[4])

        input_keys = [
            'Coal Flow (t/h)',
            'O2 Out APH (%)',
            'Corrected Flue Gas Out Temperature (°C)',
            'Feedwater temperature (℃)',
            'Feedwater flow (t/h)',
            'Energy Input From Boiler (Kcal/h)',
            'Boiler oxygen level (%)'
        ]
        optimal_inputs = dict(zip(input_keys, optimal_inputs))

        return optimal_inputs, optimal_value

    if module_name == "锅炉进口空预器":
        def objective_function(
            x): return objective_function_aph(x, model=model)
        optimal_inputs, optimal_value = pso(objective_function, lb=[b[0] for b in bounds], ub=[
                                            b[1] for b in bounds], swarmsize=particles, maxiter=iterations)

        input_keys = ['O2 in APH (%)', 'Flue Gas in Temperature (°C)',
                      'Flue gas temperature (℃)']
        optimal_inputs = dict(zip(input_keys, optimal_inputs))

        return optimal_inputs, optimal_value

    if module_name == "给水系统":
        def objective_function(
            x): return objective_function_water(x, model=model)
        optimal_inputs, optimal_value = pso(objective_function, lb=[b[0] for b in bounds], ub=[
                                            b[1] for b in bounds], swarmsize=particles, maxiter=iterations)

        input_keys = [
            'Superheater desuperheating water flow (t/h)',
            'Reheater desuperheating water flow (t/h)',
            'Feedwater pressure (MPa)',
            'Flue gas temperature (℃)',
            'Circulating water outlet temperature (℃)'
        ]
        optimal_inputs = dict(zip(input_keys, optimal_inputs))

        return optimal_inputs, optimal_value
