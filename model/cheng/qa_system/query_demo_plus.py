# -*- coding: utf-8 -*-
import spacy
import re
from neo4j import GraphDatabase
from spacy.matcher import PhraseMatcher
from sentence_transformers import SentenceTransformer, util

# TODO: 要考虑对象创建释放，数据库链接等性能优化

# 加载一个预训练的中文自然语言处理模型，使其准备好对中文文本进行分析和处理
nlp = spacy.load("zh_core_web_sm")

file_path = "triplets.txt"
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

# 数据库凭证
uri = "bolt://localhost:7687"
username = "neo4j"
password = "password"

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


def calculate_similarity(text1, text2):
    # 将文本转换为向量
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

    # 返回得分最高的三个结果
    top_results = [result for result, score in scored_results[:3]]
    return top_results


def extract_keywords(question):
    doc = nlp(question)
    # noun_phrases = []

    # 使用 PhraseMatcher 匹配自定义词表中的术语
    # 发现只用 PhraseMatcher 效果更好（考虑了词汇之间的依赖关系，更适合提取复合名词或专有名词短语）
    matches = matcher(doc)
    extracted_keywords = [
        doc[start:end].text for match_id, start, end in matches]

    # # 提取名词短语
    # for token in doc:
    #     if token.pos_ in ['NOUN', 'PROPN']:
    #         subtree_span = doc[token.left_edge.i: token.right_edge.i + 1]
    #         noun_phrases.append(subtree_span.text)

    # # 合并从PhraseMatcher和名词短语提取的关键词
    # keywords = list(set(noun_phrases + extracted_keywords))
    # return keywords

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

    with GraphDatabase.driver(uri, auth=(username, password)) as driver:
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
    print("Question:\n", question, '\n\n')

    # 提取关键词
    keywords = extract_keywords(question)
    print("Extracted Keywords:\n", keywords, '\n\n')

    # 创建 Cypher 查询
    query = create_cypher_query(keywords)
    print("Constructed Cypher Query:\n", query, '\n\n')

    # 执行查询并获取结果
    results = execute_query(query)
    print("Query Results:\n", results, '\n\n')

    return "下面是与改问题可能相关的三元组（可能包含了一些于该问题无关的三元组，请根据实际情况进行判断）: \n" + \
        str(results)


# 示例使用
if __name__ == '__main__':
    while True:
        question = input("请输入问题：")

        # 检测是否退出
        if question.lower() == 'q':
            break

        results = extract_triple(question)
        print(results)
