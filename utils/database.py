import pymysql
from dbutils.pooled_db import PooledDB
from neo4j import GraphDatabase

# MySQL database configuration
mysql_db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '123456',
    'database': 'mine'
}

# Create MySQL database pool
def create_mysql_db_pool():
    return PooledDB(
        pymysql,
        mincached=5,
        maxcached=10,
        maxconnections=20,
        blocking=True,
        **mysql_db_config
    )

# Neo4j database credentials
neo4j_uri = "bolt://localhost:7687"
neo4j_username = "neo4j"
neo4j_password = "password"

# Connect to Neo4j database
def create_neo4j_driver():
    return GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))

# MySQL database pool instance
mysql_db_pool = create_mysql_db_pool()

# Neo4j database driver instance
neo4j_driver = create_neo4j_driver()
