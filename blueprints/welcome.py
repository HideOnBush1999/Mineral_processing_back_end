from utils.websocket import socketio
import pandas as pd
import os
from flask_socketio import emit
from flask import Blueprint, request, jsonify, send_file
from minio.error import S3Error
from utils.database import get_minio_client
import pandas as pd
import io
from sklearn.ensemble import RandomForestRegressor
import joblib
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from utils.logger import logger
from utils.optimization import get_welcome_model, interpolate_data
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

welcome = Blueprint('welcome', __name__, url_prefix='/welcome')

filePath = "data/welcome/data.xlsx"
data_cache = []
streaming = False


def read_excel_data(file_path):
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        # Convert Timestamp objects to strings
        df = df.applymap(lambda x: x.isoformat()
                         if isinstance(x, pd.Timestamp) else x)
        data = df.to_dict(orient='records')
        return data
    else:
        return []


# 在程序启动时读取数据
data_cache = read_excel_data(filePath)


def stream_data():
    global data_cache, streaming
    while streaming:     # 保证当数据发送完了的时候，再从头开始发送
        for row in data_cache:
            if not streaming:
                break
            socketio.emit('update_data', row, namespace='/welcome')
            # print(row)
            socketio.sleep(3)    # 使用 socketio.sleep 不会阻塞主线程


@welcome.route('/trigger', methods=['GET'])
def trigger():
    global streaming
    if not streaming:
        streaming = True
        logger.info("Streaming started")
        socketio.start_background_task(target=stream_data)
    return jsonify({"message": "Streaming started"}), 200


@welcome.route('/stop', methods=['GET'])
def stop():
    global streaming
    streaming = False
    return jsonify({"message": "Streaming stopped"}), 200


@socketio.on('disconnect', namespace='/welcome')
def handle_disconnect():
    print('Client disconnected')


@socketio.on('connect', namespace='/welcome')
def handle_connect():
    print('Client connected')
    emit('response', {'message': 'Connected to the server'},
         namespace='/welcome')


#  ------------------------------------------------------------------------------------------
# ---------------------  下面为与抽屉界面相关的接口 --------------------------------------------
# --------------------------------------------------------------------------------------------
# 文件上传
@welcome.route('/file_upload', methods=['POST'])
def file_upload_input():
    try:
        # 确保桶存在
        minio_client = get_minio_client()
        bucket_name = 'welcome'
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

       # 检查请求中是否包含文件和文件类型字段
        if 'file' not in request.files or 'file_type' not in request.form:
            return jsonify({'error': 'No file part or file type in the request'}), 400

        file = request.files['file']
        file_type = request.form['file_type']

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
        local_dir = ''
        if file_type == 'input':
            local_dir = './data/welcome/input'
        elif file_type == 'output':
            local_dir = './data/welcome/output'
        else:
            return jsonify({'error': 'Invalid file type'}), 400

        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        with open(os.path.join(local_dir, filename), 'wb') as local_file:
            local_file.write(file_content)

        return jsonify({'message': 'File uploaded and copied successfully'}), 200

    except S3Error as e:
        return jsonify({'error': str(e)}), 500

    except Exception as e:
        return jsonify({'error': 'An error occurred during file upload'}), 500


# 在线训练
@welcome.route('/online_train', methods=['POST'])
def online_train():
    try:
        # 获取请求参数
        data = request.get_json()
        dataset_input_name = data.get('dataset_input_name')
        dataset_output_name = data.get('dataset_output_name')
        module_name = data.get('module_name')
        n_estimators = int(data.get('n_estimators'))
        max_depth = data.get('max_depth')
        min_samples_split = int(data.get('min_samples_split'))
        min_samples_leaf = int(data.get('min_samples_leaf'))

        # 训练模型
        X_train, _, y_train, _ = data_split(
            dataset_input_name, dataset_output_name, module_name)
        train_online(X_train, y_train, module_name, n_estimators,
                     max_depth, min_samples_split, min_samples_leaf)
        return jsonify({'message': '在线训练成功'}), 200

    except Exception as e:
        return jsonify({'error': f'在线训练失败: {str(e)}'}), 500


# 模型展示
@welcome.route('/model_show', methods=['POST'])
def model_show():
    try:
        # 获取请求参数
        data = request.get_json()
        dataset_input_name = data.get('dataset_input_name')
        dataset_output_name = data.get('dataset_output_name')
        module_name = data.get('module_name')
        model_name = data.get('model_name')

        # 读取模型
        model = get_welcome_model(model_name)

        # 读取数据集
        _, X_test, _, y_test = data_split(
            dataset_input_name, dataset_output_name, module_name)

        # 预测结果
        y_pred = model.predict(X_test)
        print(f"y_test type: {type(y_test)}, shape: {np.shape(y_test)}")
        print(f"y_pred type: {type(y_pred)}, shape: {np.shape(y_pred)}")

        # 计算评价指标
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # 获取样本编号（这里简单地以索引作为样本编号，从0开始，可根据实际调整）
        sample_numbers = np.arange(len(y_test))

        # 每20个点采样，通过切片操作实现，步长设置为20
        sampled_sample_numbers = sample_numbers[::20]
        sampled_y_test = y_test[::20]
        sampled_y_pred = y_pred[::20]

        # 绘制预测值与真值对比图，横坐标为样本编号，纵坐标为真值和预测值
        plt.figure(figsize=(8.33, 4))
        plt.plot(sampled_sample_numbers, sampled_y_test,
                 label='True Values', marker='o')
        plt.plot(sampled_sample_numbers, sampled_y_pred,
                 label='Predicted Values', marker='s')
        plt.xlabel('Sample Number')
        plt.ylabel('Values')
        plt.title('True vs Predicted Values Comparison')
        plt.legend()  # 添加图例，区分真值和预测值曲线

        # 将绘制的图像转换为 base64 编码的字符串，以便能在 JSON 中返回
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')

        plt.close()  # 关闭图像，避免占用过多内存

        return jsonify({'mse': mse, 'r2': r2, 'image': img_base64}), 200

    except Exception as e:
        return jsonify({'error': f'An error occurred during model showing: {str(e)}'}), 500


# 得到模型列表
@welcome.route('/get_model_list', methods=['POST'])
def get_model_list():
    try:
        # 获取请求参数
        data = request.get_json()
        module_name = data.get('module_name')

        # 从 MinIO 中读取模型列表
        minio_client = get_minio_client()
        bucket_name = 'welcome-model'
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

        objects = minio_client.list_objects(bucket_name, prefix=module_name)
        # 处理每个文件名，提取名称并构建新的格式数据
        model_list = []
        for index, obj in enumerate(objects, start=1):  # 从1开始计数作为id
            model_name = obj.object_name
            model_list.append({"id": index, "name": model_name})

        return jsonify({'model_list': model_list}), 200

    except Exception as e:
        return jsonify({'error': f'An error occurred during get model list: {str(e)}'}), 500


# TODO: 要是这里进行了关联分析的话，要重新进行判断，我觉得可以将关联分析后的结果存在 temp 文件夹上，然后在这里读取
def data_split(dataset_input_name, dataset_output_name, module_name):
    # 构建输入和输出数据集的路径
    input_path = os.path.join('./data/welcome/input', dataset_input_name)
    output_path = os.path.join('./data/welcome/output', dataset_output_name)

    logger.info("读取输入数据集：{}".format(input_path))
    try:
        # 读取输入数据集
        input_data = pd.read_excel(input_path)
    except Exception as e:
        logger.error("输入数据集读取失败：{}".format(e))

    logger.info("读取输出数据集：{}".format(output_path))
    try:
        # 读取输出数据集
        output_data = pd.read_excel(output_path)
    except Exception as e:
        logger.error("输出数据集读取失败：{}".format(e))

    try:
        # 合并输入和输出数据集
        data = pd.concat([input_data, output_data], axis=1)
        logger.info("数据集合并成功")
    except Exception as e:
        logger.error("数据集合并失败：{}".format(e))

    if module_name == "给煤机" or module_name == "给风机" or module_name == "磨煤机":
        data = data.drop(data.index[:50])
        if module_name == "给煤机":
            X = data[['皮带转速', '比例系数']]
            y = data['给煤量']

        if module_name == "给风机":
            X = data[['热风阀门开度', '冷风阀门开度', '热一次风温度', '冷一次风温度']]
            y = data[['入口一次风流量', '入口一次风温度']]

        if module_name == "磨煤机":
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


def train_online(X_train, y_train, module_name, n_estimators, max_depth, min_samples_split, min_samples_leaf):
    logger.info("开始训练模型")
    # 训练模型
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 保存模型到本地
    model_dir = './model/welcome-model'
    timeStamp = datetime.now().strftime('%Y%m%d%H%M%S')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(
        model_dir, f"{module_name}_{timeStamp}_{n_estimators}.pkl")
    joblib.dump(model, model_path)

    # 保存模型到 MinIO
    minio_client = get_minio_client()
    bucket_name = 'welcome-model'
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
    minio_client.fput_object(
        bucket_name,
        f"{module_name}_{timeStamp}_{n_estimators}.pkl",
        model_path
    )


# 删除模型
@welcome.route('/delete', methods=['POST'])
def delete_multi_layer_model():
    try:
        data = request.json
        file_name = data.get('model_name')
        minio_client = get_minio_client()
        bucket_name = 'welcome-model'

        minio_client.remove_object(bucket_name, file_name)
        return jsonify({'message': 'Model deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 下载模型
@welcome.route('/download', methods=['GET'])
def download_multi_layer_model():
    try:
        file_name = request.args.get('model_name', type=str)
        minio_client = get_minio_client()
        bucket_name = 'welcome-model'

        response = minio_client.get_object(bucket_name, file_name)
        data = response.read()
        response.close()
        response.release_conn()

        return send_file(BytesIO(data), attachment_filename=file_name, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 灰度关联分析
@welcome.route('/gray_correlation_analysis', methods=['POST'])
def gray_correlation_analysis():
    data = request.get_json()
    dataset_input_name = data.get('dataset_input_name')
    dataset_output_name = data.get('dataset_output_name')
    input_path = os.path.join('./data/welcome/input', dataset_input_name)
    output_path = os.path.join('./data/welcome/output', dataset_output_name)

    try:
        input_data = pd.read_excel(input_path)
        output_data = pd.read_excel(output_path)

        # 标准化数据
        std_input = (input_data - input_data.min()) / (input_data.max() - input_data.min())
        std_output = (output_data - output_data.min()) / (output_data.max() - output_data.min())
        # 取 std_output 的最后一列
        std_output = std_output.iloc[:, -1]

        # 初始化关联系数矩阵
        grey_matrix = np.zeros(std_input.shape)

        # 计算关联系数
        for i in range(std_input.shape[1]):
            grey_matrix[:, i] = np.abs(std_input.iloc[:, i] - std_output)

        # 计算关联度
        rho = 0.5  # 分辨系数，通常取值在0到1之间
        grey_relation = (np.min(grey_matrix) + rho * np.max(grey_matrix)) / (grey_matrix + rho * np.max(grey_matrix))
        grey_relation_degree = np.mean(grey_relation, axis=0)

        # 将特征关联度与特征名称对应（这里假设输入数据的列名就是特征名称）
        feature_names = input_data.columns
        relation_results = pd.DataFrame({'Feature': feature_names, 'RelationDegree': grey_relation_degree})

        # 生成图片相关代码
        matplotlib.rcParams['font.family'] = 'SimHei'
        plt.figure(figsize=(10, 6))
        bars = plt.barh(relation_results['Feature'], relation_results['RelationDegree'], color='skyblue')
        plt.xlabel('Relation Degree')
        plt.title('Gray Correlation Analysis Results')
        plt.gca().invert_yaxis()

        # 在每个条形图上显示对应的数值
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', 
                     va='center', ha='left', fontsize=10)

        # 将图表保存到内存中
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)

        # 将图像转换为 base64 编码（如果需要在前端以某种特定方式展示图片，可能会用到这个编码后的字符串，这里先保留转换步骤）
        img_base64 = base64.b64encode(img.getvalue()).decode('utf8')

        return jsonify({
            "message": "Gray correlation analysis completed successfully",
            "relation_results": relation_results.to_dict(orient='records'),
            "image_base64": img_base64  # 返回图片的base64编码字符串，方便前端使用
        }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
