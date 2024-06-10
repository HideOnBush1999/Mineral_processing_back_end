from flask import Blueprint, request, jsonify
from minio.error import S3Error
import os
from utils.database import get_minio_client
import pandas as pd
import io
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import base64


parameter_extraction = Blueprint(
    'parameter_extraction', __name__, url_prefix='/parameter_extraction')


# 上传文件
@parameter_extraction.route('/file_upload', methods=['POST'])
def file_upload_input():
    try:
        # 确保桶存在
        minio_client = get_minio_client()
        bucket_name = 'correlation'
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
            local_dir = './data/correlation/input'
        elif file_type == 'output':
            local_dir = './data/correlation/output'
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


# 显示文件信息
@parameter_extraction.route('/file_show', methods=['GET'])
def file_show():
    # 从请求参数中获取文件名
    filename = request.args.get('filename')
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400

    # 定义文件路径
    file_path_input = f'./data/correlation/input/{filename}'
    file_path_output = f'./data/correlation/output/{filename}'

    # 检查文件是否存在于输入或输出目录
    if os.path.exists(file_path_input):
        file_path = file_path_input
    elif os.path.exists(file_path_output):
        file_path = file_path_output
    else:
        return jsonify({'error': 'File not found'}), 404

    try:
        print(f"Reading file: {file_path}")
        # 读取 Excel 文件
        df = pd.read_excel(file_path, engine='openpyxl')

        # 获取表头列名
        column_names = df.columns.tolist()

        # 获取数据条数（不含表头）
        data_count = len(df)

        # 获取列数
        column_count = len(column_names)

        return jsonify({'column_count': column_count, 'column_names': column_names, 'data_count': data_count}), 200

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': str(e)}), 500


# 随机森林模型训练
@parameter_extraction.route('/random_forest', methods=['POST'])
def random_forest():
    data = request.get_json()
    input_file = data.get('input_file')
    output_file = data.get('output_file')
    n_estimators = data.get('n_estimators', 100)  # 设置默认值为100

    if not input_file or not output_file:
        return jsonify({"error": "Missing input_file or output_file"}), 400

    file_path_input = f'./data/correlation/input/{input_file}'
    file_path_output = f'./data/correlation/output/{output_file}'

    if not os.path.exists(file_path_input) or not os.path.exists(file_path_output):
        return jsonify({"error": "File not found"}), 404

    # 加载数据
    try:
        input_data = pd.read_excel(file_path_input)
        output_data = pd.read_excel(file_path_output)
        X = input_data
        y = output_data.iloc[:, 0]
    except Exception as e:
        return jsonify({"error": f"Error loading files: {str(e)}"}), 500

    # 训练模型
    try:
        model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=42)
        model.fit(X, y)

        # 将模型保存到本地文件
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        model_filename = f'random_forest_model_{timestamp}_{n_estimators}.pkl'
        local_dir = './model/para-extraction-rf'
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        local_path = os.path.join(local_dir, model_filename)
        joblib.dump(model, local_path)

    except Exception as e:
        return jsonify({"error": f"Error training model: {str(e)}"}), 500

    # 将模型上传到 MinIO
    try:
        minio_client = get_minio_client()
        bucket_name = 'para-extraction-rf'
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

        with open(local_path, 'rb') as model_file:
            minio_client.put_object(
                bucket_name,
                model_filename,
                model_file,
                os.path.getsize(local_path)
            )

    except S3Error as e:
        return jsonify({"error": f"Error uploading model to MinIO: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

    return jsonify({
        "message": "Model successfully trained and uploaded",
    }), 200


# 列出模型
@parameter_extraction.route('/list_models', methods=['GET'])
def list_models():
    try:
        minio_client = get_minio_client()
        bucket_name = 'para-extraction-rf'
        if not minio_client.bucket_exists(bucket_name):
            return jsonify({"error": "Bucket not found"}), 404

        objects = minio_client.list_objects(bucket_name)
        object_list = [obj.object_name for obj in objects]

        return jsonify({
            "message": "Objects retrieved successfully",
            "objects": object_list
        }), 200

    except S3Error as e:
        return jsonify({"error": f"Error listing objects in MinIO: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


# 下载模型
@parameter_extraction.route('/download_model', methods=['POST'])
def download_model():
    try:
        data = request.get_json()
        model_name = data.get('model_name')

        if not model_name:
            return jsonify({"error": "Missing model_name in request"}), 400

        # 检查本地文件是否存在
        local_dir = './model/para-extraction-rf'
        local_path = os.path.join(local_dir, model_name)
        if os.path.exists(local_path):
            return jsonify({
                "message": "Model already exists locally",
                "local_path": local_path
            }), 200

        minio_client = get_minio_client()
        bucket_name = 'para-extraction-rf'
        if not minio_client.bucket_exists(bucket_name):
            return jsonify({"error": "Bucket not found"}), 404

        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        minio_client.fget_object(bucket_name, model_name, local_path)

        return jsonify({
            "message": "Model downloaded successfully",
            "local_path": local_path
        }), 200

    except S3Error as e:
        return jsonify({"error": f"Error downloading model from MinIO: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


# 内置特征重要性函数
@parameter_extraction.route('/feature_importance', methods=['POST'])
def feature_importance():
    data = request.get_json()
    model_name = data.get('model_name')
    input_file = data.get('input_file')
    output_file = data.get('output_file')
    number_key_parameters = data.get('number_key_parameters', None)

    model_path = f'./model/para-extraction-rf/{model_name}'
    if not os.path.exists(model_path):
        return jsonify({"error": "Model not found"}), 404
    model = joblib.load(model_path)

    file_path_input = f'./data/correlation/input/{input_file}'
    if not os.path.exists(file_path_input):
        return jsonify({"error": "File not found"}), 404
    
    file_path_output = f'./data/correlation/output/{output_file}'
    if not os.path.exists(file_path_output):
        return jsonify({"error": "File not found"}), 404

    try:
        X = pd.read_excel(file_path_input)
        y = pd.read_excel(file_path_output).iloc[:, 0]
        feature_importances = model.feature_importances_
        feature_names = X.columns.tolist()
        feature_importance_dict = {
            feature_names[i]: feature_importances[i] for i in range(len(feature_names))}

        if number_key_parameters is not None:
            sorted_feature_importances = dict(
                sorted(feature_importance_dict.items(),
                       key=lambda item: item[1], reverse=True)
            )
            feature_importance_dict = dict(list(sorted_feature_importances.items())[
                                           :number_key_parameters])

        # 提取关键参数数据
        key_parameters = list(feature_importance_dict.keys())
        df_key_params = X[key_parameters]
        df_key_params = pd.concat([y, df_key_params], axis=1)

        # 生成 Excel 文件
        output_file = io.BytesIO()
        df_key_params.to_excel(output_file, index=False)
        output_file.seek(0)

        # 上传到 MinIO 桶中
        minio_client = get_minio_client()
        bucket_name = 'extracted-data'
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f'feature_importance_key_parameters_{timestamp}.xlsx'

        minio_client.put_object(
            bucket_name,
            filename,
            output_file,
            len(output_file.getvalue()),
            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

        return jsonify({
            "message": "Feature importances retrieved successfully",
            "feature_importance_dict": feature_importance_dict
        }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


# 排列特征重要性函数
@parameter_extraction.route('/permutation_importance', methods=['POST'])
def permutation_importances():
    data = request.get_json()
    model_name = data.get('model_name')
    input_file = data.get('input_file')
    output_file = data.get('output_file')
    number_key_parameters = data.get('number_key_parameters', None)

    model_path = f'./model/para-extraction-rf/{model_name}'
    if not os.path.exists(model_path):
        return jsonify({"error": "Model not found"}), 404
    model = joblib.load(model_path)

    file_path_input = f'./data/correlation/input/{input_file}'
    if not os.path.exists(file_path_input):
        return jsonify({"error": "File not found"}), 404

    file_path_output = f'./data/correlation/output/{output_file}'
    if not os.path.exists(file_path_output):
        return jsonify({"error": "File not found"}), 404

    try:
        X = pd.read_excel(file_path_input)
        y = pd.read_excel(file_path_output).iloc[:, 0]
        result = permutation_importance(
            model, X, y, n_repeats=5, random_state=42, n_jobs=2)
        sorted_idx = result.importances_mean.argsort()
        sorted_names = X.columns[sorted_idx].tolist()
        sorted_importances = result.importances_mean[sorted_idx].tolist()
        permutation_importance_dict = {
            sorted_names[i]: sorted_importances[i] for i in range(len(sorted_names))}

        if number_key_parameters is not None:
            sorted_permutation_importance_dict = dict(
                list(permutation_importance_dict.items())[
                    :number_key_parameters]
            )
            permutation_importance_dict = sorted_permutation_importance_dict

         # 提取关键参数数据
        key_parameters = list(permutation_importance_dict.keys())
        df_key_params = X[key_parameters]
        df_key_params = pd.concat([y, df_key_params], axis=1)

        # 生成 Excel 文件
        output_file_io = io.BytesIO()
        df_key_params.to_excel(output_file_io, index=False)
        output_file_io.seek(0)

        # 上传到 MinIO 桶中
        minio_client = get_minio_client()
        bucket_name = 'extracted-data'
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f'permutation_importance_key_parameters_{timestamp}.xlsx'

        minio_client.put_object(
            bucket_name,
            filename,
            output_file_io,
            len(output_file_io.getvalue()),
            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

        return jsonify({
            "message": "Permutation importances retrieved successfully",
            "permutation_importance_dict": permutation_importance_dict
        }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


# TODO: 递归特征消除函数  --> 未完成，不做也行
@parameter_extraction.route('/recursive_feature_elimination', methods=['POST'])
def recursive_feature_elimination():
    data = request.get_json()
    model_name = data.get('model_name')  # 模型名称
    input_file = data.get('input_file')  # 输入文件名
    output_file = data.get('output_file')  # 输出文件名

    print(data)
    model_path = f'./model/para-extraction-rf/{model_name}'
    if not os.path.exists(model_path):
        return jsonify({"error": "Model not found"}), 404
    model = joblib.load(model_path)

    file_path_input = f'./data/correlation/input/{input_file}'
    if not os.path.exists(file_path_input):
        return jsonify({"error": "File not found"}), 404

    file_path_output = f'./data/correlation/output/{output_file}'
    if not os.path.exists(file_path_output):
        return jsonify({"error": "File not found"}), 404

    try:
        X = pd.read_excel(file_path_input)
        y = pd.read_excel(file_path_output).iloc[:, 0]
        rfe = RFE(model, n_features_to_select=10, step=1)
        print("abc")
        rfe.fit(X, y)
        selected_names = X.columns[rfe.get_support()].tolist()
        print(selected_names)

        return jsonify({
            "message": "Recursive feature elimination selected features successfully",
            "selected_names": selected_names
        }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500



@parameter_extraction.route('/model_evaluation', methods=['POST'])
def model_evaluation():
    data = request.get_json()
    model_name = data.get('model_name')
    input_file = data.get('input_file')
    output_file = data.get('output_file')

    model_path = f'./model/para-extraction-rf/{model_name}'
    if not os.path.exists(model_path):
        return jsonify({"error": "Model not found"}), 404
    model = joblib.load(model_path)

    file_path_input = f'./data/correlation/input/{input_file}'
    if not os.path.exists(file_path_input):
        return jsonify({"error": "File not found"}), 404

    file_path_output = f'./data/correlation/output/{output_file}'
    if not os.path.exists(file_path_output):
        return jsonify({"error": "File not found"}), 404
    
    try:
        X = pd.read_excel(file_path_input)
        y = pd.read_excel(file_path_output).iloc[:, 0]
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        # Sampling every 50th data point
        sample_indices = list(range(0, len(y), 50))
        y_sampled = y.iloc[sample_indices]
        y_pred_sampled = y_pred[sample_indices]

        # Plotting the actual vs predicted results with sampling
        plt.figure(figsize=(10, 6))
        plt.plot(sample_indices, y_sampled, label='Actual', marker='o')
        plt.plot(sample_indices, y_pred_sampled, label='Predicted', marker='x')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Actual vs Predicted')
        plt.legend()

        # Save the plot to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='png')
        img_io.seek(0)
        img_base64 = base64.b64encode(img_io.getvalue()).decode()

        return jsonify({
            "mse": mse,
            "r2": r2,
            "image": img_base64
        }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500