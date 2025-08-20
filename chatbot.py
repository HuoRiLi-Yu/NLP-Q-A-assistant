import os
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from zhipuai import ZhipuAI
from flask import Flask, request, jsonify
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)  

def read_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def preprocess_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def extract_keywords(text, top_n=10):
    vectorizer = TfidfVectorizer(max_features=top_n)
    X = vectorizer.fit_transform([text])
    keywords = vectorizer.get_feature_names_out()
    return keywords

def get_sentence_embeddings(text, nlp):
    doc = nlp(text)
    return np.mean([token.vector for token in doc if token.has_vector], axis=0)

def auto_run_conversation(question, context):
    api_key = '817ccfdb8584415fbf7a38aca72a4048.vbdjDVSUZ00dhGDG'
    client = ZhipuAI(api_key=api_key) 
    # 构建模型提示信息
    system_prompt = f"以下是关于大语言模型的详细描述：\n\n{context}\n\n请回答该问题，在我提供的文本中，直接使用文本内容回答问题，不要有除了文本内容之外的扩展回答和思考，直截了当地给出答案即可，并且所有答案都合并为一个段落，不需要分点答。"
    response = client.chat.completions.create(
        model="glm-4",  
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": question}]
    )
    final_response = response.choices[0].message.content
    return final_response

# 加载spaCy模型
nlp = spacy.load('en_core_web_sm')

# 读取和预处理内容
content_path = r'D:\NLP作业\NLP期末大作业\大语言模型.txt'
context = read_content(content_path)
context = preprocess_text(context)

# 提取关键词
keywords = extract_keywords(context)
#print(f"提取的关键词: {keywords}")

# 获取句向量
context_embedding = get_sentence_embeddings(context, nlp).reshape(1, -1)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    text = data.get("question")
    try:
        # 预处理问题
        question = preprocess_text("请根据我给出的文本信息，回答："+text)
        question_embedding = get_sentence_embeddings(question, nlp).reshape(1, -1)

        # 计算相似度并找到最相似的上下文段落
        similarity = cosine_similarity(context_embedding, question_embedding)
        most_similar_index = np.argmax(similarity)
        most_similar_context = context
        # print(most_similar_context)
        response = auto_run_conversation(question, most_similar_context)
        return jsonify({'responseText': response})
    except Exception as e:
        print(e)
        return jsonify({'responseText': '聊天机器人未能执行，请您检查一下后台。'})

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000)
