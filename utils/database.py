# database.py
import pymysql
from dbutils.pooled_db import PooledDB

db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '123456',
    'database': 'mine'
}


def create_db_pool():
    return PooledDB(
        pymysql,
        mincached=5,
        maxcached=10,
        maxconnections=20,
        blocking=True,
        **db_config
    )


db_pool = create_db_pool()
