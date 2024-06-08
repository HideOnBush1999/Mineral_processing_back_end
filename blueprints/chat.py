# -*- coding: utf-8 -*-
from flask import Blueprint, request, jsonify
from xinference.client import Client
from utils.database import get_neo4j_driver
from utils.logger import logger
import spacy
import re
from spacy.matcher import PhraseMatcher
from sentence_transformers import SentenceTransformer, util
from utils.websocket import socketio
import json

# 全局定义变量
client = Client("http://localhost:9997")
model_uid = None
model = None


nlp = None
matcher = None
model_extract = None


def get_sentence_transformer_model():
    global model_extract
    if model_extract is None:
        model_extract = SentenceTransformer(
            'paraphrase-multilingual-MiniLM-L12-v2')
    return model_extract


# 加载一个预训练的中文自然语言处理模型，使其准备好对中文文本进行分析和处理
def get_nlp_model():
    global nlp
    if nlp is None:
        nlp = spacy.load("zh_core_web_sm")
    return nlp


def get_matcher():
    global matcher
    if matcher is None:
        nlp = get_nlp_model()

        file_path = "E:\\myData\\study\\研究生\\项目\\矿产加工系统\\back\\data\\triplets.txt"
        with open(file_path, 'r', encoding='utf-8') as file:
            triplets = file.readlines()

        # 提取三元组中的实体和关系
        entities_and_relations = set()

        # 正则表达式匹配
        pattern = re.compile(r'\(([^,]+), ([^,]+)(?:, ([^,]+))?\)')

        for triplet in triplets:
            match = pattern.search(triplet)
            if match:
                # 将匹配到的实体和关系添加到集合中
                entities_and_relations.update(match.groups())

        # 将集合转换为列表并排序
        custom_wordlist = sorted(list(entities_and_relations))

        # 创建 PhraseMatcher
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        patterns = [nlp.make_doc(word) for word in custom_wordlist]
        matcher.add("CustomTerms", patterns)
    return matcher


qa = Blueprint('qa', __name__, url_prefix='/qa')


@qa.route('/chat', methods=['POST'])
def chat():
    try:
        prompt = request.json.get('prompt')
        logger.info(f"收到聊天请求，提示：{prompt}")

        # 提取三元组
        triple = extract_triple(prompt)

        if not triple:
            response = "未找到相关度较高的三元组"
        else:
            response = "查询到相关的三元组为：\n"
            for result, score in triple:
                response += f"（{result[0]}, {result[1]}, {result[2]}） 相关度：{score:.2f}\n"
            
        # 调用后台任务启动对话流
        data = {'prompt': prompt, 'triple': response}
        logger.info(f"启动后台任务，数据：{json.dumps(data)}")
        socketio.start_background_task(target=start_chat_stream_background, data=data)

        return jsonify({"message": response.strip()})

    except Exception as e:
        logger.error(f"聊天错误：{e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


def start_chat_stream_background(data):
    logger.info(f"后台任务收到数据：{data}")
    global model
    prompt = data.get('prompt')
    triple = data.get('triple')

    # 对 chat_history 进行初始化，原先的 chat_history 被清空，所以不支持多轮对话
    chat_history = [{
        'role': 'assistant',
        'content': triple
    }]
    try:
        # 调用模型进行聊天，添加 stream=True 后，返回的是一个迭代器
        response_stream = model.chat(
            prompt=prompt, chat_history=chat_history, generate_config={"max_tokens": 1024, "stream": True})

        # 逐步处理流式响应
        for response in response_stream:
            delta = response['choices'][0].get('delta', {})
            content = delta.get('content', '')  # 提取实际的内容
            logger.info(f"发送 chat_response 内容：{content}")
            socketio.emit('chat_response', {'content': content}, namespace='/qa')

        socketio.emit('chat_response', {'content': "#finish#"}, namespace='/qa')
        logger.info(f"对话流结束")

    except Exception as e:
        logger.error(f"聊天流错误：{e}")
        socketio.emit('chat_response_complete', {'full_response': "对话结束，对话流异常"}, namespace='/qa')


#  客户端可以调用该接口启动对话流，但是前端并没有调用该接口，因为在 chat 接口中已经启动了后台任务
@socketio.on('start_chat_stream', namespace='/qa')
def start_chat_stream(data):
    logger.info(f"收到 start_chat_stream 事件，数据：{data}")
    start_chat_stream_background(data)


@qa.route('/create_model', methods=['POST'])
def create_model():
    global client, model_uid, model
    try:
        model_name = "local-qwen-7b-q5_k_m"
        model_uid = client.launch_model(model_name=model_name)
        model = client.get_model(model_uid)
        logger.info("Model created successfully")
        return jsonify({
            'status': 'success',
            'message': f'Model {model_name} created with UID {model_uid}'
        }), 200
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error creating model: {e}'
        }), 500


@qa.route('/terminate_model', methods=['POST'])
def terminate_model():
    try:
        global client, model_uid, model
        client.terminate_model(model_uid)
        model_uid = None
        model = None
        logger.info("Model terminated successfully")
        return jsonify({
            'status': 'success',
            'message': f'Model terminated'
        }), 200
    except Exception as e:
        logger.error(f"Error while terminating model: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error while terminating model: {e}'
        }), 500


def calculate_similarity(text1, text2):
    # 将文本转换为向量
    model = get_sentence_transformer_model()
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)

    # 计算余弦相似度
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    return cosine_scores.item()


def filter_results(results, question):
    # 计算每个结果的相似度得分
    scores = []
    for result in results:
        # 将三元组转换为文本
        triplet_text = f"{result[0]} {result[1]} {result[2]}"
        # 计算与原始问题的相似度
        similarity = calculate_similarity(question, triplet_text)
        scores.append(similarity)

    # 将结果和得分打包在一起，然后根据得分排序
    scored_results = sorted(zip(results, scores),
                            key=lambda x: x[1], reverse=True)

    # 过滤得分大于 0.1 的结果，并取前三高
    top_results = [(result, score) for result, score in scored_results if score > 0.1][:3]
    return top_results


def extract_keywords(question):
    nlp = get_nlp_model()
    matcher = get_matcher()

    doc = nlp(question)

    # 使用 PhraseMatcher 匹配自定义词表中的术语
    # 发现只用 PhraseMatcher 效果更好（考虑了词汇之间的依赖关系，更适合提取复合名词或专有名词短语）
    matches = matcher(doc)
    extracted_keywords = [
        doc[start:end].text for match_id, start, end in matches]

    return extracted_keywords


def create_cypher_query(keywords):
    if not keywords:  # 检查关键词列表是否为空
        return None  # 或者返回一个适当的消息，例如 "没有提取到关键词"

    # 构造基于关键词的模糊匹配查询
    # 假设所有关键词都是实体的一部分
    keyword_conditions = ["e1.name =~ '.*" + keyword +
                          ".*' OR e2.name =~ '.*" + keyword + ".*'" for keyword in keywords]

    query = "MATCH (e1)-[r]->(e2) WHERE " + \
        " OR ".join(keyword_conditions) + " RETURN e1, r, e2"
    return query


def execute_query(query):
    if query is None:
        return []

    neo4j_driver = get_neo4j_driver()
    with neo4j_driver as driver:
        with driver.session() as session:
            result = session.run(query)
            formatted_results = []
            for record in result:
                e1_name = record['e1']['name']
                relation_type = record['r'].type
                e2_name = record['e2']['name']
                formatted_results.append((e1_name, relation_type, e2_name))
            return formatted_results


def extract_triple(question):
    logger.info(f"Question:\n{question}\n\n")

    # 提取关键词
    keywords = extract_keywords(question)
    logger.info(f"Extracted Keywords:\n{keywords}\n\n")

    # 创建 Cypher 查询
    query = create_cypher_query(keywords)
    logger.info(f"Constructed Cypher Query:\n{query}\n\n")

    # 执行查询并获取结果
    results = execute_query(query)
    logger.info(f"Query Results:\n{results}\n\n")

    # 过滤结果并返回最相关的三元组
    results = filter_results(results, question)
    logger.info(f"Filtered Results:\n{results}\n\n")

    return results
