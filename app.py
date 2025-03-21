from flask import Flask
from flask_cors import CORS
from blueprints.login import login
from blueprints.welcome import welcome
from blueprints.chat import qa
from blueprints.traid import traid
from blueprints.multi_layer_modeling import multi_layer_modeling
from blueprints.multi_layer_manage import multi_layer_manage
from blueprints.parameter_extraction import parameter_extraction
from blueprints.correlation_analysis import correlation_analysis
from utils.websocket import socketio
from utils.celery import celery_config

app = Flask(__name__)
CORS(app)
app.config.update(celery_config)
socketio.init_app(app, cors_allowed_origins="*", async_mode='eventlet')


app.register_blueprint(login)
app.register_blueprint(welcome)
app.register_blueprint(multi_layer_modeling)
app.register_blueprint(multi_layer_manage)
app.register_blueprint(qa)
app.register_blueprint(traid)
app.register_blueprint(parameter_extraction)
app.register_blueprint(correlation_analysis)


if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=5005, debug=False)
