# 提供三元组的分页展示、查询、新增、修改、删除功能
# 其中，新增、修改、删除功能需要具有管理员权限

from flask import request, jsonify, Blueprint
from utils.database import neo4j_driver

traid = Blueprint('traid', __name__, url_prefix='/traid')


@traid.route('/get_traids', methods=['GET'])
def get_traids():
    page = request.args.get('page', default=1, type=int)
    limit = request.args.get('limit', default=10, type=int)
    skip = (page - 1) * limit

    with neo4j_driver.session() as session:
        result = session.run(
            "MATCH (s)-[r]->(o) RETURN s, type(r) as relation, o SKIP $skip LIMIT $limit",
            skip=skip, limit=limit
        )
        triples = []
        for record in result:
            triples.append({
                'subject': record['s']['name'],
                'relation': record['relation'],
                'object': record['o']['name']
            })
        return jsonify(triples)


@traid.route('/search', methods=['GET'])
def get_traid():
    keyword = request.args.get('keyword', '', type=str)

    with neo4j_driver.session() as session:
        result = session.run(
            """
            MATCH (s)-[r]->(o) 
            WHERE s.name CONTAINS $keyword OR o.name CONTAINS $keyword OR type(r) CONTAINS $keyword 
            RETURN s, type(r) as relation, o
            """,
            keyword=keyword
        )
        triples = []
        for record in result:
            triples.append({
                'subject': record['s']['name'],
                'relation': record['relation'],
                'object': record['o']['name']
            })
        return jsonify(triples)


@traid.route('/add', methods=['POST'])
def add_traid():
    data = request.json
    subject = data.get('subject')
    relation = data.get('relation')
    object_ = data.get('object')

    if not subject or not relation or not object_:
        return jsonify({'error': 'Missing required parameters'}), 400

    with neo4j_driver.session() as session:
        try:
            # MERGE 语句确保如果节点或关系已经存在，它们将不会被创建新的，而是使用现有的
            result = session.run(
                "MERGE (s:Entity {name: $subject}) "
                "MERGE (o:Entity {name: $object}) "
                "MERGE (s)-[r:`" + relation + "`]->(o)",
                subject=subject, object=object_
            )
            # Check if the query was successful
            if result.consume().counters.nodes_created > 0 or result.consume().counters.relationships_created > 0:
                return jsonify({'message': 'Triple added successfully'}), 201
            else:
                return jsonify({'message': 'No changes made to the database'}), 200
        except Exception as e:
            print(f"Error adding triple: {e}")
            return jsonify({'error': 'Failed to add triple'}), 500


@traid.route('/update', methods=['PUT'])
def update_traid():
    data = request.json
    old_subject = data.get('old_subject')
    old_relation = data.get('old_relation')
    old_object = data.get('old_object')
    new_subject = data.get('new_subject')
    new_relation = data.get('new_relation')
    new_object = data.get('new_object')

    if not all([old_subject, old_relation, old_object, new_subject, new_relation, new_object]):
        return jsonify({'error': 'Missing required parameters'}), 400

    with neo4j_driver.session() as session:
        try:
            with session.begin_transaction() as tx:
                # 删除旧关系
                tx.run(
                    "MATCH (s:Entity {name: $old_subject})-[r:`" +
                    old_relation + "`]->(o:Entity {name: $old_object}) "
                    "DELETE r",
                    old_subject=old_subject, old_object=old_object
                )
                # 更新节点名称
                tx.run(
                    "MATCH (s:Entity {name: $old_subject}) "
                    "SET s.name = $new_subject",
                    old_subject=old_subject, new_subject=new_subject
                )
                tx.run(
                    "MATCH (o:Entity {name: $old_object}) "
                    "SET o.name = $new_object",
                    old_object=old_object, new_object=new_object
                )
                # 创建新关系
                tx.run(
                    "MATCH (s:Entity {name: $new_subject}), (o:Entity {name: $new_object}) "
                    "CREATE (s)-[r:`" + new_relation + "`]->(o)",
                    new_subject=new_subject, new_object=new_object
                )
            return jsonify({'message': 'Triple updated successfully'}), 200
        except Exception as e:
            print(f"Error updating triple: {e}")
            return jsonify({'error': 'Failed to update triple'}), 500


# 直接更新节点名称可能会影响与该节点相关的其他关系。
# 这种情况下，我们应该避免直接修改节点名称
# 删除旧关系：首先匹配旧的关系，并将其删除
# 创建新节点（如有必要）：如果新旧节点名称不同，创建新的节点
# 创建新关系：匹配新的节点，并创建新的关系
@traid.route('/update', methods=['PUT'])
def update_traid():
    data = request.json
    old_subject = data.get('old_subject')
    old_relation = data.get('old_relation')
    old_object = data.get('old_object')
    new_subject = data.get('new_subject')
    new_relation = data.get('new_relation')
    new_object = data.get('new_object')

    if not all([old_subject, old_relation, old_object, new_subject, new_relation, new_object]):
        return jsonify({'error': 'Missing required parameters'}), 400

    with neo4j_driver.session() as session:
        try:
            with session.begin_transaction() as tx:
                # 删除旧关系
                tx.run(
                    "MATCH (s:Entity {name: $old_subject})-[r:`" +
                    old_relation + "`]->(o:Entity {name: $old_object}) "
                    "DELETE r",
                    old_subject=old_subject, old_object=old_object
                )
                # 创建新节点（如果名称不同）
                if old_subject != new_subject:
                    tx.run(
                        "MERGE (s:Entity {name: $new_subject})",
                        new_subject=new_subject
                    )
                if old_object != new_object:
                    tx.run(
                        "MERGE (o:Entity {name: $new_object})",
                        new_object=new_object
                    )
                # 创建新关系
                tx.run(
                    "MATCH (s:Entity {name: $new_subject}), (o:Entity {name: $new_object}) "
                    "CREATE (s)-[r:`" + new_relation + "`]->(o)",
                    new_subject=new_subject, new_object=new_object
                )
            return jsonify({'message': 'Triple updated successfully'}), 200
        except Exception as e:
            print(f"Error updating triple: {e}")
            return jsonify({'error': 'Failed to update triple'}), 500


# 只删除一条边，不删除节点
@traid.route('/delete', methods=['DELETE'])
def delete_traid():
    data = request.json
    subject = data.get('subject')
    relation = data.get('relation')
    object_ = data.get('object')

    if not subject or not relation or not object_:
        return jsonify({'error': 'Missing required parameters'}), 400

    with neo4j_driver.session() as session:
        try:
            result = session.run(
                "MATCH (s:Entity {name: $subject})-[r:`" + relation +
                "`]->(o:Entity {name: $object}) DELETE r",
                subject=subject, object=object_
            )
            if result.consume().counters.relationships_deleted > 0:
                return jsonify({'message': 'Triple deleted successfully'}), 200
            else:
                return jsonify({'message': 'No matching triple found'}), 404
        except Exception as e:
            print(f"Error deleting triple: {e}")
            return jsonify({'error': 'Failed to delete triple'}), 500


# 删除一条边，并删除节点，这是危险操作，因为节点删除了，其他和这个节点相关的关系也全都被删除了
# @traid.route('/delete_all', methods=['DELETE'])
# def delete_all_traid():
#     data = request.json
#     subject = data.get('subject')
#     relation = data.get('relation')
#     object_ = data.get('object')

#     if not subject or not relation or not object_:
#         return jsonify({'error': 'Missing required parameters'}), 400

#     with neo4j_driver.session() as session:
#         try:
#             query = (
#                 "MATCH (s:Entity {name: $subject})-[r:`" + relation + "`]->(o:Entity {name: $object}) "
#                 "DELETE r, s, o"
#             )
#             print(f"Running query: {query} with subject={subject}, relation={relation}, object={object_}")
#             result = session.run(query, subject=subject, object=object_)
#             summary = result.consume()
#             if summary.counters.nodes_deleted > 0 and summary.counters.relationships_deleted > 0:
#                 return jsonify({'message': 'Triple and nodes deleted successfully'}), 200
#             else:
#                 return jsonify({'message': 'No matching triple found'}), 404
#         except Exception as e:
#             print(f"Error deleting triple and nodes: {e}")
#             return jsonify({'error': 'Failed to delete triple and nodes'}), 500
