from flask import Blueprint, request, jsonify, send_file
from utils.database import get_minio_client
from io import BytesIO

multi_layer_manage = Blueprint(
    'multi_layer_manage', __name__, url_prefix='/multi_layer_manage')

# 多层模型的管理  实现多层模型的增删改查  模型本身存放在 minio 服务器上

# 分页展示  （除了展示名字外，还需要展示模型的大小）
@multi_layer_manage.route('/get_multi_layer_models', methods=['GET'])
def get_multi_layer_models():
    try:
        page = request.args.get('page', default=1, type=int)
        limit = request.args.get('limit', default=10, type=int)
        skip = (page - 1) * limit

        minio_client = get_minio_client()
        bucket_name = 'multi-layer-model'
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

        objects = list(minio_client.list_objects(bucket_name, recursive=True))  # 转换为列表
        total = len(objects)  # 计算总数
        objects_list = objects[skip:skip+limit]  # 获取分页后的对象
        models = []
        for obj in objects_list:
            models.append({
                'name': obj.object_name,
                'size': f"{round(obj.size / (1024 * 1024), 2)} MB"
            })

        # 返回数据不满 limit 的时候，补充空数据
        for _ in range(limit - len(models)):
            models.append({'name': '', 'size': ''})

        return jsonify({'models': models, 'total': total})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 增
@multi_layer_manage.route('/add', methods=['POST'])
def add_multi_layer_model():
    try:
        file = request.files['file']
        file_name = file.filename
        file.seek(0, 2)  # 移动到文件末尾
        file_size = file.tell()  # 获取文件大小
        file.seek(0)  # 移动回文件开头

        minio_client = get_minio_client()
        bucket_name = 'multi-layer-model'
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

        minio_client.put_object(bucket_name, file_name, file, file_size)
        return jsonify({'message': 'Model added successfully'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 删
@multi_layer_manage.route('/delete', methods=['POST'])
def delete_multi_layer_model():
    try:
        data = request.json
        file_name = data.get('model_name')
        minio_client = get_minio_client()
        bucket_name = 'multi-layer-model'

        minio_client.remove_object(bucket_name, file_name)
        return jsonify({'message': 'Model deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 改  (只是修改名称)
@multi_layer_manage.route('/update', methods=['POST'])
def update_multi_layer_model():
    try:
        data = request.json
        old_name = data.get('old_name')
        new_name = data.get('new_name')
        minio_client = get_minio_client()
        bucket_name = 'multi-layer-model'

        # Download the old object
        response = minio_client.get_object(bucket_name, old_name)
        data = response.read()
        response.close()
        response.release_conn()

        # Upload the data with new name
        minio_client.put_object(bucket_name, new_name, BytesIO(data), len(data))

        # Remove the old object
        minio_client.remove_object(bucket_name, old_name)

        return jsonify({'message': 'Model updated successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 查
@multi_layer_manage.route('/search', methods=['GET'])
def search_multi_layer_model():
    try:
        query = request.args.get('query', default='', type=str)
        minio_client = get_minio_client()
        bucket_name = 'multi-layer-model'
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

        objects = minio_client.list_objects(bucket_name, recursive=True)
        models = []
        for obj in objects:
            if query in obj.object_name:
                models.append({
                    'name': obj.object_name,
                    # Convert size to MB and append "MB" unit
                    'size': f"{round(obj.size / (1024 * 1024), 2)} MB"
                })
        total = len(models)

        return jsonify({'models': models, 'total': total})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 下载模型
@multi_layer_manage.route('/download', methods=['GET'])
def download_multi_layer_model():
    try:
        file_name = request.args.get('model_name', type=str)
        minio_client = get_minio_client()
        bucket_name = 'multi-layer-model'

        response = minio_client.get_object(bucket_name, file_name)
        data = response.read()
        response.close()
        response.release_conn()

        return send_file(BytesIO(data), attachment_filename=file_name, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
