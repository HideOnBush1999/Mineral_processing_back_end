from flask import Blueprint, request, jsonify
from minio.error import S3Error
import os
from utils.database import get_minio_client
import numpy as np
import pandas as pd
import io
from sklearn.ensemble import RandomForestRegressor
import joblib
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from utils.logger import logger
from utils.optimization import get_optimization_results



multi_layer_modeling = Blueprint(
    'multi_layer_modeling', __name__, url_prefix='/multi_layer_modeling')


# 上传文件
@multi_layer_modeling.route('/file_upload', methods=['POST'])
def file_upload_input():
    try:
        # 确保桶存在
        minio_client = get_minio_client()
        bucket_name = 'multi-layer-data'
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

       # 检查请求中是否包含文件和文件类型字段
        if 'file' not in request.files:
            return jsonify({'error': 'No file part or file type in the request'}), 400

        file = request.files['file']

        # 如果用户没有选择文件，浏览器提交的文件名可能为空
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # 读取文件内容到内存中
        file_content = file.read()

        # 使用安全的文件名，并上传文件到MinIO
        filename = file.filename
        minio_client.put_object(
            bucket_name,
            filename,
            io.BytesIO(file_content),
            len(file_content)
        )

        # 复制文件到本地目录
        local_dir = './data/multi_layer'
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        with open(os.path.join(local_dir, filename), 'wb') as local_file:
            local_file.write(file_content)

        return jsonify({'message': 'File uploaded and copied successfully'}), 200

    except S3Error as e:
        return jsonify({'error': str(e)}), 500

    except Exception as e:
        return jsonify({'error': f'An error occurred during file upload: {str(e)}'}), 500


# 在线训练  （请求： 数据集名， 模块名， 决策树数量）
@multi_layer_modeling.route('/online_train', methods=['POST'])
def online_train():
    try:
        # 获取请求参数
        data = request.get_json()
        dataset_name = data.get('dataset_name')
        module_name = data.get('module_name')
        tree_num = int(data.get('tree_num'))

        # 训练模型
        X_train, _, y_train, _ = data_split(dataset_name, module_name)
        train_online(X_train, y_train, module_name, tree_num)
        return jsonify({'message': 'Online training completed successfully'}), 200

    except Exception as e:
        return jsonify({'error': f'An error occurred during online training: {str(e)}'}), 500


# 模型展示  （请求：数据集名， 模块名, 模型名）
@multi_layer_modeling.route('/model_show', methods=['POST'])
def model_show():
    try:
        # 获取请求参数
        data = request.get_json()
        dataset_name = data.get('dataset_name')
        module_name = data.get('module_name')
        model_name = data.get('model_name')

        # 读取模型
        model = get_model(model_name)

        # 读取数据集
        _, X_test, _, y_test = data_split(dataset_name, module_name)

        # 预测结果
        y_pred = model.predict(X_test)

        # 计算评价指标
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # TODO： 缺少绘图，绘图每个模型返回的图片数量不一样，会有点麻烦
        return jsonify({'mse': mse, 'r2': r2}), 200

    except Exception as e:
        return jsonify({'error': f'An error occurred during model showing: {str(e)}'}), 500


# 获取模型列表  （请求: 模块名）
@multi_layer_modeling.route('/get_model_list', methods=['POST'])
def get_model_list():
    try:
        # 获取请求参数
        data = request.get_json()
        module_name = data.get('module_name')

        # 从 MinIO 中读取模型列表
        minio_client = get_minio_client()
        bucket_name = 'multi-layer-model'
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

        objects = minio_client.list_objects(bucket_name, prefix=module_name)
        model_list = [obj.object_name for obj in objects]

        return jsonify({'model_list': model_list}), 200

    except Exception as e:
        return jsonify({'error': f'An error occurred during get model list: {str(e)}'}), 500


# 优化求解  (请求：数据集名， 模块名， 模型名)
@multi_layer_modeling.route('/optimization_solve', methods=['POST'])
def optimization_solve():
    try:
        # 获取请求参数
        data = request.get_json()
        dataset_name = data.get('dataset_name')
        module_name = data.get('module_name')
        model_name = data.get('model_name')

        model = get_model(model_name)

        # 得到对应参数的上下限
        bounds = get_bounds(dataset_name, module_name)

        # 得到求解的结果
        optimal_inputs, optimal_value = get_optimization_results(dataset_name, module_name, model, bounds, particles=30, iterations=2)
        
        # 将 NumPy 数组转换为 Python 列表
        optimal_inputs_list = optimal_inputs.tolist()
        optimal_value_list = optimal_value.tolist()

        logger.info(f"optimal_inputs: {optimal_inputs_list}, optimal_value: {optimal_value_list}".format())
        return jsonify({'optimal_inputs': optimal_inputs_list, 'optimal_value': optimal_value_list}), 200

    except Exception as e:
        return jsonify({'error': f'An error occurred during optimization: {str(e)}'}), 500


def interpolate_data(df, num_points):
    x_old = np.arange(len(df))
    x_new = np.linspace(0, len(df) - 1, num=num_points)
    df_interpolated = pd.DataFrame()

    for col in df.columns:
        f = interp1d(x_old, df[col], kind='quadratic',
                     fill_value="extrapolate")
        df_interpolated[col] = f(x_new)

    return df_interpolated


def data_split(dataset_name, module_name):
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
            X = data[['皮带转速', '比例系数']]
            y = data['给煤量']

        if module_name == "给风机":
            X = data[['热风阀门开度', '冷风阀门开度', '热一次风温度', '冷一次风温度']]
            y = data[['入口一次风流量', '入口一次风温度']]

        if module_name == "磨煤":
            X = data[['入口一次风流量', '入口一次风温度', '给煤量', '磨煤机电流', '原煤温度', '原煤水分']]
            y = data['出口煤粉流量']

    if module_name == "锅炉进口空预器" or module_name == "给水系统" or module_name == "锅炉燃烧":
        data = data.dropna()
        num_samples = 1000
        data = interpolate_data(data, num_samples)
        logger.info("数据预处理完成")

        if module_name == "锅炉进口空预器":
            X = data[['O2 in APH (%)', 'Flue Gas in Temperature (°C)',
                     'Flue gas temperature (℃)']]
            y = data[['O2 Out APH (%)',
                     'Corrected Flue Gas Out Temperature (°C)']]

        if module_name == "给水系统":
            X = data[[
                'Superheater desuperheating water flow (t/h)',
                'Reheater desuperheating water flow (t/h)',
                'Feedwater pressure (MPa)',
                'Flue gas temperature (℃)',
                'Circulating water outlet temperature (℃)'
            ]]
            y = data[['Feedwater temperature (℃)', 'Feedwater flow (t/h)']]

        if module_name == "锅炉燃烧":
            X = data[[
                'Coal Flow (t/h)',
                'O2 Out APH (%)',
                'Corrected Flue Gas Out Temperature (°C)',
                'Feedwater temperature (℃)',
                'Feedwater flow (t/h)',
                'Energy Input From Boiler (Kcal/h)',
                'Boiler oxygen level (%)'
            ]]
            y = data[[
                'Boiler Eff (%)',
                'SO2 (mg/m3)',
                'Nox (mg/m3)',
                'CO (mg/m3)',
                'CO2 (ppm)',
                'Main steam temperature (boiler side) (℃)',
                'Main steam pressure (boiler side) (Mpa)'
            ]]
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def train_online(X_train, y_train, module_name, tree_num):
    logger.info("开始训练模型")
    # 训练模型
    model = RandomForestRegressor(n_estimators=tree_num, random_state=42)
    model.fit(X_train, y_train)

    # 保存模型到本地
    model_dir = './model/multi-layer-model'
    timeStamp = datetime.now().strftime('%Y%m%d%H%M%S')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(
        model_dir, module_name + '_' + timeStamp + '_' + str(tree_num) + '.pkl')
    joblib.dump(model, model_path)

    # 保存模型到 MinIO
    minio_client = get_minio_client()
    bucket_name = 'multi-layer-model'
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
    minio_client.fput_object(
        bucket_name,
        module_name + '_' + timeStamp + '.pkl',
        model_path
    )


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
