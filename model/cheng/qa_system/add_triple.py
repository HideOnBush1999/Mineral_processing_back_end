"""
如果节点和关系不存在: MERGE 会创建新的节点和关系。
如果节点和关系已经存在: MERGE 不会创建新的实体，而是使用现有的实体。这意味着它不会重复添加已经存在的节点和关系。
"""

from neo4j import GraphDatabase

# 数据库凭证
uri = "bolt://localhost:7687"
username = "neo4j"
password = "password"

# 连接到数据库
driver = GraphDatabase.driver(uri, auth=(username, password))

def add_triple(tx, entity1, relation, entity2):
    # 确保关系名称是有效的标识符
    relation = relation.replace('"', '').replace(' ', '_')
    # Cypher 查询创建节点和关系
    query = (
        f"MERGE (e1:Entity {{name: '{entity1}'}}) "
        f"MERGE (e2:Entity {{name: '{entity2}'}}) "
        f"MERGE (e1)-[:{relation}]->(e2)"
    )
    tx.run(query)

# 处理文件并将三元组添加到 Neo4j 的函数
def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file, driver.session() as session:
        count = 0
        for line in file:
            # 使用strip()移除行尾的空格和换行符
            cleaned_line = line.strip()
            if cleaned_line.startswith('(') and cleaned_line.endswith(')'):
                triple = cleaned_line[1:-1].split(', ')
                if len(triple) == 3:
                    entity1, relation, entity2 = triple
                    session.write_transaction(add_triple, entity1, relation, entity2)
                    print(entity1, relation, entity2)
                    count += 1
        print(f"Processed {count} triples")

# 处理三元组文件
process_file('./三元组.txt')

# 关闭数据库连接
driver.close()
