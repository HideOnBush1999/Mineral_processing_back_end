from flask import Flask
from flask_cors import CORS
from blueprints.login import login
from blueprints.chat import qa
from blueprints.traid import traid
from blueprints.multi_layer_modeling import multi_layer_modeling
from blueprints.correlation_analysis import correlation_analysis

app = Flask(__name__)
CORS(app)


app.register_blueprint(login)
app.register_blueprint(qa)
app.register_blueprint(traid)
app.register_blueprint(multi_layer_modeling)
app.register_blueprint(correlation_analysis)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5005, debug=True)
