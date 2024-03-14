from flask import Flask, jsonify
from flask_cors import CORS
from utils.logger import logger
from blueprints.login import login
from blueprints.qa_system import qa
from blueprints.multi_layer_modeling import multi_layer_modeling
from blueprints.correlation_analysis import correlation_analysis
# from xinference.client import Client

app = Flask(__name__)
CORS(app)


app.register_blueprint(login)
app.register_blueprint(qa)
app.register_blueprint(multi_layer_modeling)
app.register_blueprint(correlation_analysis)

# model = None

# @app.before_first_request
# def load_models():
#     global model
#     client = Client("http://localhost:9997")
#     model_uid = client.launch_model(model_name="local-qwen-7b-q5_k_m")
#     model = client.get_model(model_uid)

@app.route('/hello', methods=['GET'])
def hello():
    logger.debug('这是一个 debug 级别的日志')
    logger.info('这是一个 info 级别的日志')
    return jsonify({'message': 'Hello World!'}), 200


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5005, debug=False)
