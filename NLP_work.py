import jieba
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
import re
from collections import defaultdict


def load_local_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def load_stopwords(file_paths):
    stopwords = set()
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                stopwords.add(line.strip())
    return stopwords

def preprocess_text(text, stopwords):
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = jieba.lcut(text, cut_all=False)
    words = [word for word in words if word not in stopwords and word.strip()]
    return ' '.join(words)

def split_sentences(text):
    sentences = re.split(r'[。！？；：,!?:;\n\r]', text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences

# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 加载BERT模型和分词器
try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertModel.from_pretrained('bert-base-chinese')
    print("Tokenizer and model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please check your network connection or try the manual download method.")
    exit()

def extract_features(texts):
    if not texts:
        return np.array([])
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()


# 评分与匹配函数
def find_best_sentence_index(processed_question, processed_sentences, sentence_vectors, stopwords, word_weight=0.5, bert_weight=0.8):
    """
    计算问题与所有候选句子的加权分数，并返回得分最高句子的索引。
    这个函数现在只处理 "处理后" 的数据，以保证比较的公平性。
    """
    if not processed_sentences:
        return -1, {}

    question_vector = extract_features([processed_question])[0]
    question_words = set([word for word in jieba.lcut(processed_question) if word not in stopwords])
    
    scores = {}
    
    for i, sentence in enumerate(processed_sentences):
        # 1. 计算关键词分数（归一化）
        sentence_words = set(jieba.lcut(sentence))
        common_words = question_words.intersection(sentence_words)
        # 归一化：重合词数 / 问题总词数，避免长问题得分过高
        word_score = len(common_words) / len(question_words) if len(question_words) > 0 else 0
        
        # 2. 计算BERT语义相似度分数
        sentence_vector = sentence_vectors[i]
        bert_score = cosine_similarity([question_vector], [sentence_vector])[0][0]
        
        # 3. 加权计算总分数
        # 确保bert_score是正数，余弦相似度范围是-1到1，通常语义相似为正
        combined_score = word_weight * word_score + bert_weight * max(0, bert_score)
        scores[i] = combined_score
    
    # 找到分数最高的句子的索引
    if not scores:
        return -1, {}
        
    best_index = max(scores, key=scores.get)
    return best_index, scores

# 主函数
def main():
    # 1. 加载数据
    file_path = r"D:\NLP作业\NLP期末大作业\大语言模型.txt"  
    content = load_local_text(file_path)
    
    stop_words_files = [r'D:\NLP作业\NLP期末大作业\cn_stopwords.txt', r'D:\NLP作业\NLP期末大作业\hit_stopwords.txt', r'D:\NLP作业\NLP期末大作业\scu_stopwords.txt']
    stopwords = load_stopwords(stop_words_files)
    
    # 2. 文本处理：同时保留原始句子和处理后的句子，并保持一一对应
    original_sentences = split_sentences(content)
    processed_sentences = []
    # 建立一个临时的原始句子列表，用于同步过滤
    synced_original_sentences = []

    for sentence in original_sentences:
        processed = preprocess_text(sentence, stopwords)
        # 如果处理后句子不为空（不是纯停用词或标点），则保留
        if processed:
            processed_sentences.append(processed)
            synced_original_sentences.append(sentence)
    
    # 使用同步过滤后的原始句子列表
    original_sentences = synced_original_sentences

    if not original_sentences:
        print("文档为空或未能抽取出有效句子。")
        return

    # 3. 特征提取：只对处理后的句子进行
    print("Extracting features from sentences, please wait...")
    sentence_vectors = extract_features(processed_sentences)
    print("Feature extraction complete.")

    # 4. 问答循环
    while True:
        question = input("\n请输入您的问题：")
        if question.lower() in ['退出', 'exit']:
            break

        # 预处理问题
        processed_question = preprocess_text(question, stopwords)
        if not processed_question:
            print("您的问题经过处理后为空，请换个问法。")
            continue

        # 查找最佳答案
        best_index, all_scores = find_best_sentence_index(
            processed_question,
            processed_sentences,
            sentence_vectors,
            stopwords,
            word_weight=0.4, # 降低关键词权重
            bert_weight=0.6  # 提高语义权重
        )
        
        if best_index != -1:
            # 从原始句子列表中，根据索引找到最匹配的原文并输出
            best_sentence = original_sentences[best_index]
            print("答：", best_sentence)
        else:
            print("抱歉，我在文档中没有找到相关答案。")

if __name__ == "__main__":
    main()