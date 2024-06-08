from flask import Flask
from flask_cors import CORS
from blueprints.login import login
from blueprints.chat import qa
from blueprints.traid import traid
from blueprints.multi_layer_modeling import multi_layer_modeling
from blueprints.parameter_extraction import parameter_extraction
from blueprints.correlation_analysis import correlation_analysis
from utils.websocket import socketio

app = Flask(__name__)
CORS(app)
socketio.init_app(app, cors_allowed_origins="*")


app.register_blueprint(login)
app.register_blueprint(qa)
app.register_blueprint(traid)
app.register_blueprint(multi_layer_modeling)
app.register_blueprint(parameter_extraction)
app.register_blueprint(correlation_analysis)


if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=5005, debug=False)
