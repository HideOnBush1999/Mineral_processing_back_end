from flask import Blueprint, request, jsonify
from utils.encryption import verify_password
from utils.database import db_pool
from utils.logger import logger


login = Blueprint('login', __name__, url_prefix='/login')


@login.route('/', methods=['POST'])
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
                    if username == 'admin':
                        logger.info(f'admin用户 {username} 登录成功')
                        # 返回指定的内容
                        return jsonify({
                            'success': True,
                            'data': {
                                'username': username,
                                'roles': ['admin'],
                                'accessToken': 'eyJhbGciOiJIUzUxMiJ9.admin',
                                'refreshToken': 'eyJhbGciOiJIUzUxMiJ9.adminRefresh',
                                'expires': '2023/10/30 00:00:00'
                            }
                        }), 200
                    else:
                        logger.info(f'user用户 {username} 登录成功')
                        return jsonify({
                            'success': True,
                            'data': {
                                'username': username,
                                'roles': ['user'],
                                'accessToken': 'eyJhbGciOiJIUzUxMiJ9.user',
                                'refreshToken': 'eyJhbGciOiJIUzUxMiJ9.userRefresh',
                                'expires': '2023/10/30 00:00:00'
                            }
                        }), 200
                else:
                    logger.info(f'user用户 {username} 登录失败')
                    return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
    except Exception as e:
        print("An error occurred:", e)
        return jsonify({'message': 'An error occurred'}), 500
