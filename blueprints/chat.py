# -*- coding: utf-8 -*-
from flask import Blueprint, request, jsonify
from xinference.client import Client
from utils.database import neo4j_driver
from utils.logger import logger
import spacy
import re
from spacy.matcher import PhraseMatcher
from sentence_transformers import SentenceTransformer, util

# 全局定义变量
client = Client("http://localhost:9997")
model_uid = None
model = None

# 加载一个预训练的中文自然语言处理模型，使其准备好对中文文本进行分析和处理
nlp = spacy.load("zh_core_web_sm")

file_path = "E:\\myData\study\\研究生\\项目\\矿产加工系统\\back\\data\\triplets.txt"
with open(file_path, 'r', encoding='utf-8') as file:
    triplets = file.readlines()

# 提取三元组中的实体和关系
entities_and_relations = set()

# 使用正则表达式来解析每行中的实体和关系
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


model_extract = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


qa = Blueprint('qa', __name__, url_prefix='/qa')


@qa.route('/chat', methods=['POST'])
def chat():
    global model
    try:
        prompt = request.json.get('prompt')
        logger.info(f"Received chat request with prompt: {prompt}")

        # 提取三元组
        triple = extract_triple(prompt)

        chat_history = [{
            'role': 'assistant',
            'content': triple
        }]

        # 调用模型进行聊天
        response = model.chat(
            prompt=prompt, chat_history=chat_history, generate_config={"max_tokens": 1024})

        content = response['choices'][0]['message']['content']
        logger.info(f"chat_history: {chat_history}, response: {content}")
        return jsonify({
            'status': 'success',
            'chat_history': chat_history,
            'content': content
        }), 200
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


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
    embeddings1 = model_extract.encode(text1, convert_to_tensor=True)
    embeddings2 = model_extract.encode(text2, convert_to_tensor=True)

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

    # 返回得分最高的三个结果
    top_results = [result for result, score in scored_results[:3]]
    return top_results


def extract_keywords(question):
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

    results = filter_results(results, question)
    logger.info(f"Filtered Results:\n{results}\n\n")

    return "下面是与改问题可能相关的三元组（可能包含了一些于该问题无关的三元组，请根据实际情况进行判断）: \n" + \
        str(results)
