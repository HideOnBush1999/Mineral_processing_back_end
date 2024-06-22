from flask import Blueprint, request

multi_layer_manage = Blueprint('multi_layer_manage', __name__, url_prefix='/multi_layer_manage')

# 多层模型的管理  实现多层模型的增删改查  模型本身存放在 minio 服务器上

# 分页展示  （除了展示名字外，还需要展示模型的大小）
@multi_layer_manage.route('/get_multi_layer_models', methods=['GET'])
def get_multi_layer_models():
    pass

# 增
@multi_layer_manage.route('/add', methods=['POST'])
def add_multi_layer_model():
    pass


# 删
@multi_layer_manage.route('/delete', methods=['POST'])
def delete_multi_layer_model():
    pass


# 改  (只是修改名称)
@multi_layer_manage.route('/update', methods=['POST'])
def update_multi_layer_model():
    pass


# 查
@multi_layer_manage.route('/search', methods=['GET'])
def search_multi_layer_model():
    pass


# 下载模型
@multi_layer_manage.route('/download', methods=['GET'])
def download_multi_layer_model():
    pass