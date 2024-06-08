import pymysql
from dbutils.pooled_db import PooledDB
from neo4j import GraphDatabase
from minio import Minio

# 全局变量
mysql_db_pool = None
neo4j_driver = None
minio_client = None

# MySQL database configuration
mysql_db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '123456',
    'database': 'mine'
}

# Create MySQL database pool
def get_mysql_db_pool():
    global mysql_db_pool
    if mysql_db_pool is None:
        mysql_db_pool = PooledDB(
            pymysql,
            mincached=5,
            maxcached=10,
            maxconnections=20,
            blocking=True,
            **mysql_db_config
        )
    return mysql_db_pool

# Neo4j database credentials
neo4j_uri = "bolt://localhost:7687"
neo4j_username = "neo4j"
neo4j_password = "password"

# Connect to Neo4j database
def get_neo4j_driver():
    global neo4j_driver
    if neo4j_driver is None:
        neo4j_driver = GraphDatabase.driver(
            neo4j_uri, auth=(neo4j_username, neo4j_password))
    return neo4j_driver

# MinIO client configuration
minio_config = {
    'endpoint': '127.0.0.1:9005',
    'access_key': 'minioadmin',
    'secret_key': 'minioadmin',
    'secure': False  # 如果使用的是http而不是https，设为False
}

# Create MinIO client
def get_minio_client():
    global minio_client
    if minio_client is None:
        minio_client = Minio(
            minio_config['endpoint'],
            access_key=minio_config['access_key'],
            secret_key=minio_config['secret_key'],
            secure=minio_config['secure']
        )
    return minio_client

