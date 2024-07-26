from flask import Blueprint, jsonify
from utils.websocket import socketio
import pandas as pd
import os
import time
from flask_socketio import emit
from utils.logger import logging

welcome = Blueprint('welcome', __name__, url_prefix='/welcome')

filePath = "data/welcome/data.xlsx"
data_cache = []
streaming = False

def read_excel_data(file_path):
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        # Convert Timestamp objects to strings
        df = df.applymap(lambda x: x.isoformat() if isinstance(x, pd.Timestamp) else x)
        data = df.to_dict(orient='records')
        return data
    else:
        return []

# 在程序启动时读取数据
data_cache = read_excel_data(filePath)


def stream_data():
    global data_cache, streaming
    while streaming:     # 保证当数据发送完了的时候，再从头开始发送
        for row in data_cache:
            if not streaming:
                break
            socketio.emit('update_data', row, namespace='/welcome')
            # print(row)
            socketio.sleep(3)    # 使用 socketio.sleep 不会阻塞主线程


@welcome.route('/trigger', methods=['GET'])
def trigger():
    global streaming
    if not streaming:
        streaming = True
        logging.info("Streaming started")
        socketio.start_background_task(target=stream_data)
    return jsonify({"message": "Streaming started"}), 200


@welcome.route('/stop', methods=['GET'])
def stop():
    global streaming
    streaming = False
    return jsonify({"message": "Streaming stopped"}), 200


@socketio.on('disconnect', namespace='/welcome')
def handle_disconnect():
    print('Client disconnected')


@socketio.on('connect', namespace='/welcome')
def handle_connect():
    print('Client connected')
    emit('response', {'message': 'Connected to the server'}, namespace='/welcome')
