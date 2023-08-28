from flask import Flask, request, jsonify
from flask_cors import CORS
import pymysql
from dbutils.pooled_db import PooledDB

from utils.encryption import verify_password

app = Flask(__name__)
CORS(app)

# MySQL数据库连接池配置
db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '123456',
    'database': 'mine'
}

db_pool = PooledDB(
    pymysql,          # 第一个参数是数据库连接驱动，这里使用 pymysql
    mincached=5,      # 最小闲置连接数，即在池中保持的最少连接数
    maxcached=10,     # 最大闲置连接数，即在池中保持的最大连接数
    maxconnections=20,  # 最大连接数，包括最大闲置连接数和正在使用的连接数
    blocking=True,    # 当池中没有可用连接时，是否阻塞等待新连接
    **db_config       # 其他数据库连接参数，包括主机、用户名、密码、数据库名称等
)


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    try:
        with db_pool.connection() as conn:   # 使用数据库连接池中的连接
            with conn.cursor() as cursor:    # 连接上创建游标
                query = "SELECT password, salt FROM users WHERE username = %s"   # 查询语句
                cursor.execute(query, (username,))  # 执行查询语句
                result = cursor.fetchone()          # 获取一条结果
                hashed_password = result[0].encode()
                salt = result[1].encode()

                if result and verify_password(password, hashed_password, salt):
                    return jsonify({'message': 'Login successful'}), 200
                else:
                    return jsonify({'message': 'Invalid credentials'}), 401
    except Exception as e:
        print("An error occurred:", e)
        return jsonify({'message': 'An error occurred'}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5005, debug=True)
