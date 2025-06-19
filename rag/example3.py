from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np


model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 示例文档集合
documents = [
    {"text": "人工智能是通过模拟人类智能来完成特定任务的技术。", "category": "技术"},
    {"text": "机器学习是人工智能的一个子领域，主要关注通过数据训练模型。", "category": "技术"},
    {"text": "深度学习是一种利用神经网络进行数据处理的机器学习方法。", "category": "深度学习"},
    {"text": "自然语言处理使得计算机能够理解和生成人类语言。", "category": "自然语言处理"},
    {"text": "数据科学帮助分析复杂数据以及从中提取有价值的信息。", "category": "数据科学"},
]


def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings




documents_embeddings = np.vstack([embed_text(doc["text"]) for doc in documents])
dimension = documents_embeddings.shape[1]
# hidden的维度
print(dimension)

index = faiss.IndexFlatL2(dimension)
index.add(documents_embeddings.astype('float32'))

def search(query, top_k=2):
    query_embedding = embed_text(query)
    distances, indices = index.search(query_embedding.astype('float32'), top_k)
    return [(documents[i] ,distances[0][j]) for j , i in enumerate(indices[0])]


def search_with_filter(query,category=None, top_k=2):
    query_embedding = embed_text(query)
    distances, indices = index.search(query_embedding.astype('float32'), top_k * 2)

    filtered_results = []
    for idx in indices[0]:
        doc = documents[idx]
        if category is None or doc["category"] == category:
            filtered_results.append((doc["text"], doc["category"]))
            if len(filtered_results) == top_k:
                break
    return filtered_results

query_text = "如何通过数据分析来提高效率？"
result = search_with_filter(query_text,category="数据科学")

for text,category in result:
    print(f"文档: {text}")
    print(f"类别: {category}")
    print("================================")