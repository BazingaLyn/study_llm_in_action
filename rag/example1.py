import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer


model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

#示例文档集合
documents=[
    "人工智能是通过模拟人类智能来完成特定任务的技术。",
    "机器学习是人工智能的一个子领域，主要关注通过数据训练模型。",
    "深度学习是一种利用神经网络进行数据处理的机器学习方法。",
    "自然语言处理使得计算机能够理解和生成人类语言。"
]

vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents).toarray()

dimension = doc_vectors.shape[1]
print(f"向量维度: {dimension}")

index = faiss.IndexFlatL2(dimension)  # 使用L2距离作为相似度度量
index.add(doc_vectors.astype('float32')) # 将向量添加到索引中

def retrieve_documents(query, top_k=1):
    query_vector = vectorizer.transform([query]).toarray().astype('float32')
    distances, indices = index.search(query_vector, top_k)
    return [documents[i] for i in indices[0]]

def generate_response(query):
    documents = retrieve_documents(query)
    context = " ".join(documents)
    inputs = tokenizer.encode(context +" "+ query, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, temperature=0.7, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


query = "什么是机器学习？"
response = generate_response(query)
print(f"问题: {query}")
print(f"回答: {response}")
print("================================")