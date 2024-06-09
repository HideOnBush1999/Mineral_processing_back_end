from flask import Blueprint, request

multi_layer_manage = Blueprint('multi_layer_manage', __name__, url_prefix='/multi_layer_manage')

# 多层模型的管理  实现多层模型的增删改查  模型本身存放在 minio 服务器上
