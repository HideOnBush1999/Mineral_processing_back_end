### 后端（Flask）命名规范

1. **变量命名：** 同样使用驼峰命名法，但是首字母小写。例如：`userData`, `requestParameters`。
2. **函数命名：** 使用下划线分隔的小写字母来命名函数，以动词开头描述函数的操作。例如：`get_user_data()`, `update_user_profile()`。
3. **模块命名：** 使用下划线分隔的小写字母来命名模块文件。例如：`user_management.py`, `data_processing.py`。
4. **路由命名：** 使用下划线分隔的小写字母来命名路由，描述其功能。例如：`/get_user_data`, `/update_profile`。
5. **数据库表命名：** 使用下划线分隔的小写字母，表名通常使用复数形式。例如：`users`, `products`。
6. **常量命名：** 全部大写，多个单词间用下划线分隔。例如：`MAX_ATTEMPTS`, `DEFAULT_TIMEOUT`。
7. **数据库字段命名：** 使用下划线分隔的小写字母，保持与表名和实际数据结构的一致性。例如：`first_name`, `created_at`。
8. **文件夹命名：** 文件夹命名可以遵循类似模块文件的规则，使用下划线分隔的小写字母来命名文件夹，描述其包含内容的特点。例如：`user_management`, `data_processing`。
9. **数据库指令命名：** 关键字都使用大写。例如：`SELECT password, salt FROM users WHERE username = %s`



### 数据库

用户名: root

密码: 123456

数据库名称： mine

选定的字符集为: utf8mb4

选定的排序规则为: utf8mb4_general_ci



创建数据库

```SQL
CREATE DATABASE mine;

USE mine;

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    password VARCHAR(255) NOT NULL,
    salt VARCHAR(100) NOT NULL,
);

INSERT INTO users (username, password, salt) VALUES
('admin', '$2b$12$3c2Nkuon.78tQMhY1eLhWujVLIfPYZpRb0kU/9LDSPpjFuG4g2VDa', '$2b$12$3c2Nkuon.78tQMhY1eLhWu'),
('cheng', '$2b$12$r6.pPWaHOLONR3bxrRGFLu30jxsbdRYnP/bpHovjkLqjIK3.XnxBC', '$2b$12$r6.pPWaHOLONR3bxrRGFLu');
```

minio 对象存储

单例   资源管理

redis 存储中间状态

Websocket 加 流式推送
流式推送（如逐字逐句地发送大语言模型的回答）最适合使用 WebSocket。这是因为 WebSocket 提供了全双工通信，允许服务器实时地向客户端发送数据，而不需要客户端不断地发起新的请求。

Celery 异步任务队列  -->  用在模型的训练耗时任务上  和 go 语言中的 Asynq 库类似，Asynq 使用 Redis 作为消息代理
Celery 是一个基于 Python 的异步任务队列，它可以轻松地将耗时的任务异步化，并通过消息队列（如 RabbitMQ、Redis 等）将任务结果返回给客户端

在 Celery 的任务函数中，我是想在函数最后使用 websocket 发送消息，但是发现无法发送，可能是因为 Celery 的任务函数运行在单独的进程中，而 Flask-SocketIO 的 Socket.IO 实例只能在主进程中运行，而task = optimize_task.apply_async(args=[dataset_name, module_name, model_name]) 这个参数中也不能穿上下文进去，所以无法发送消息。
然后我想到现将结果通过redis作为中间人，先将结果发布到通道中，然后起一个进程监听这个通道，当收到消息再发送出去，但是这样还是失败。这里面应该还是上下文的问题，这里 `with app.app_context():` 是最初对应的上下文，而不是那个时刻的上下文，使用 current_app 应该也不能解决这个问题，因为一开始进入后就是下面的死循环， current_app 不会更新。  目前是新增了一个查询任务结果的接口，虽然不能主动放回结果，但是用户也可以通过这个接口查询到优化结果。
现在想到的一个方法，我应该新建另一个程序，专门监听redis的消息，然后将结果通过普通的websocket的库发送出去。


app.py

```python
from flask import Flask
from flask_cors import CORS
from blueprints.login import login
from blueprints.chat import qa
from blueprints.traid import traid
from blueprints.multi_layer_modeling import multi_layer_modeling
from blueprints.multi_layer_manage import multi_layer_manage
from blueprints.parameter_extraction import parameter_extraction
from blueprints.correlation_analysis import correlation_analysis
from utils.websocket import socketio
from utils.celery import celery_config
from utils.database import get_redis_client
from threading import Thread
from flask_socketio import emit
from multiprocessing import Process

app = Flask(__name__)
CORS(app)
app.config.update(celery_config)
socketio.init_app(app, cors_allowed_origins="*", async_mode='eventlet')


app.register_blueprint(login)
app.register_blueprint(multi_layer_modeling)
app.register_blueprint(multi_layer_manage)
app.register_blueprint(qa)
app.register_blueprint(traid)
app.register_blueprint(parameter_extraction)
app.register_blueprint(correlation_analysis)


def redis_listener():
    print("启动 Redis 监听器")
    redis_client = get_redis_client()
    pubsub = redis_client.pubsub()
    pubsub.subscribe('optimization_result')

    with app.app_context():
        for message in pubsub.listen():
            if message['type'] == 'message':
                data = message['data']
                print(data)
                print("wjj")
                socketio.emit('optimization_result', data)
                socketio.emit('optimization_result', {'hello': 'world'})

                # 下面这两种发送方式都失败
                # emit('optimization_result', {'data': 'tesk'})
                # with app.test_request_context('/'):
                #     socketio.emit('test', {'data': 'tesk'})
                # socketio.start_background_task(emit_websocket_message, data)
                print("发送 WebSocket 事件成功")


    
def start_redis_listener():
    print("启动 Redis 监听器1")
    thread = Thread(target=redis_listener)
    thread.daemon = True
    thread.start()


if __name__ == '__main__':
    redis_process = start_redis_listener()
    socketio.run(app, host='127.0.0.1', port=5005, debug=True)
    redis_process.terminate()
```


utils/celery.py

```python
from celery import Celery
from utils.optimization import get_optimization_results, get_model, get_bounds
from utils.logger import logger
from utils.websocket import socketio
from utils.database import get_redis_client
import json
from threading import Thread
from flask import current_app
from flask_socketio import emit


class CelerySingleton:
    _instance = None

    def __new__(cls, config=None):
        if cls._instance is None:
            if config is None:
                raise ValueError("A config dictionary must be provided for the first initialization.")
            cls._instance = cls._create_celery(config)
        return cls._instance

    @staticmethod
    def _create_celery(config):
        celery = Celery(
            config['app_name'],
            backend=config['result_backend'],  # 使用新格式
            broker=config['broker_url']        # 使用新格式
        )
        celery.conf.update(config)  # 确保其他配置也是使用新格式
        celery.conf.task_time_limit = 3600  # 设置任务时间限制为 3600 秒
        celery.autodiscover_tasks(['utils.celery'])  # 自动发现任务
        return celery

def make_celery(config):
    return CelerySingleton(config)

# 配置变量，使用新格式
celery_config = {
    'app_name': 'my_app',
    'broker_url': 'redis://localhost:6379/0',  # 新格式
    'result_backend': 'redis://localhost:6379/0',  # 新格式
    'broker_connection_retry_on_startup': True,
}

# 获取 Celery 实例
celery = make_celery(celery_config)


@celery.task(bind=True)
def optimize_task(self, dataset_name, module_name, model_name):
    try:
        print("开始优化任务")
        logger.info(f"开始优化任务: dataset_name={dataset_name}, module_name={module_name}, model_name={model_name}")
        model = get_model(model_name)
        bounds = get_bounds(dataset_name, module_name)
        optimal_inputs, optimal_value = get_optimization_results(dataset_name, module_name, model, bounds, particles=1, iterations=1)
        result = {'optimal_inputs': optimal_inputs.tolist(), 'optimal_value': optimal_value.tolist()}
        logger.info("优化任务完成")

        redis_client = get_redis_client()
        redis_client.publish('optimization_result', json.dumps(result))

        # 任务完成后发送WebSocket事件
        # 下面两种都发送不出去
        # socketio.emit('tesk', {'data': 'tesk'})
        # emit('tesk', {'data': 'tesk'})
        return result

    except Exception as e:
        logger.error(f"优化任务出错: {e}")

```


