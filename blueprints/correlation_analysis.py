from flask import Blueprint, request, jsonify
import base64
from flask import send_file
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from utils.database import get_minio_client
import pandas as pd
import io
import os

correlation_analysis = Blueprint('correlation_analysis', __name__, url_prefix='/correlation_analysis')


# 上传 excel
@correlation_analysis.route('/upload_excel', methods=['POST'])
def excel_file_upload():
    try:
        # 确保桶存在
        minio_client = get_minio_client()
        bucket_name = 'extracted-data'
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file provided'}), 400
        
        # 如果用户没有选择文件，浏览器提交的文件名可能为空
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # 读取文件内容到内存中
        file_name = file.filename
        file_content = file.read()
        
         # 将文件上传到 Minio
        minio_client.put_object(
            bucket_name, 
            file_name, 
            io.BytesIO(file_content),
            len(file_content)
        ) 
        
        # 复制文件到本地目录
        local_dir = './data/correlation/extracted_data'
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        
        file_path = os.path.join(local_dir, file_name)
        file.save(file_path)
        
        return jsonify({'message': 'File uploaded successfully', 'file_path': file_path}), 200
    except Exception as e:
        return jsonify({'error': 'An error occurred during file upload', 'details': str(e)}), 500

# 列出 excel
@correlation_analysis.route('/list_excel', methods=['GET'])
def list_excel():
    try:
        minio_client = get_minio_client()
        bucket_name = 'extracted-data'

        if not minio_client.bucket_exists(bucket_name):
            return jsonify({"error": "Bucket not found"}), 404

        objects = minio_client.list_objects(bucket_name)
        object_list = [obj.object_name for obj in objects]

        return jsonify({
            "message": "Objects listed successfully",
            "object_list": object_list
        }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


# 下载 excel
@correlation_analysis.route('/download_excel', methods=['POST'])
def download_excel():
    data = request.get_json()
    excel_name = data.get('excel_name')

    try:
        minio_client = get_minio_client()
        bucket_name = 'extracted-data'
        download_path = f'./data/correlation/extracted_data/{excel_name}'

        if not minio_client.bucket_exists(bucket_name):
            return jsonify({"error": "Bucket not found"}), 404

        minio_client.fget_object(bucket_name, excel_name, download_path)

        return jsonify({
            "message": "Excel downloaded successfully",
            "download_path": download_path
        }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


# 灰度关联分析
@correlation_analysis.route('/gray_correlation_analysis', methods=['POST'])
def gray_correlation_analysis():
    data = request.get_json()
    file = data.get('file')

    file_path = f'./data/correlation/extracted_data/{file}'
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    try:
        df = pd.read_excel(file_path)

        X = df.iloc[:, 1:]  # 输入特征：第二列到最后一列
        y = df.iloc[:, 0]   # 目标变量：第一列
        # 标准化数据
        std_X = (X - X.min()) / (X.max() - X.min())
        std_y = (y - y.min()) / (y.max() - y.min())

        # 初始化关联系数矩阵
        grey_matrix = np.zeros(std_X.shape)

        # 计算关联系数
        for i in range(std_X.shape[1]):
            grey_matrix[:, i] = np.abs(std_X.iloc[:, i] - std_y)

        # 计算关联度
        rho = 0.5  # 分辨系数，通常取值在0到1之间
        grey_relation = (np.min(grey_matrix) + rho * np.max(grey_matrix)) / (grey_matrix + rho * np.max(grey_matrix))
        grey_relation_degree = np.mean(grey_relation, axis=0)


        # 将特征关联度与特征名称对应
        feature_names = X.columns
        relation_results = pd.DataFrame({'Feature': feature_names, 'RelationDegree': grey_relation_degree})

        # 打印特征关联度
        print(relation_results.sort_values(by='RelationDegree', ascending=False))

        return jsonify({
            "message": "Gray correlation analysis completed successfully",
            "relation_results": relation_results.to_dict(orient='records')
        }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


# 生成对应图片
@correlation_analysis.route('/generate_image', methods=['POST'])
def generate_image():
    data = request.get_json()
    relation_results = data.get('relation_results')

    if not relation_results:
        return jsonify({"error": "No relation results provided"}), 400

    try:
        # 将数据转换为 DataFrame
        df = pd.DataFrame(relation_results)
        
        # 按 RelationDegree 排序
        df = df.sort_values(by='RelationDegree', ascending=False)

        # 创建图表
        matplotlib.rcParams['font.family'] = 'SimHei'
        plt.figure(figsize=(10, 6))
        bars = plt.barh(df['Feature'], df['RelationDegree'], color='skyblue')
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

        # # 将图像转换为 base64 编码
        # img_base64 = base64.b64encode(img.getvalue()).decode('utf8')

        # 发送图像文件
        return send_file(img, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500