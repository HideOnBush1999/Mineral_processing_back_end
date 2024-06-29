from celery import Celery
from utils.optimization import get_optimization_results, get_model, get_bounds
from utils.logger import logger


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
    'broker_url': 'redis://localhost:6379/0',      # 新格式
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

        return result

    except Exception as e:
        logger.error(f"优化任务出错: {e}")

