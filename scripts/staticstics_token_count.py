import json
from transformers import AutoTokenizer
from tqdm import tqdm

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained('../model/')

# 读取JSONL文件
file_path = '../dataset/pretrain_hq_v7.jsonl'

# 统计变量
total_tokens = 0
total_samples = 0

# 处理JSONL文件
with open(file_path, 'r', encoding='utf-8') as file:
    # 使用tqdm显示进度
    for line in tqdm(file):
        total_samples += 1
        data = json.loads(line)
        
        # 获取'text'字段
        text = data['text']
        
        # Tokenize文本
        tokens = tokenizer(text)['input_ids']
        num_tokens = len(tokens)
        
        # 累加token数量
        total_tokens += num_tokens

# 打印统计结果
tokens_in_millions = total_tokens / 1_000_000 / 1000.0
print(f"总样本数: {total_samples}")
print(f"总Token数: {total_tokens:,} ({tokens_in_millions:.2f}B)")
print(f"平均每个样本的Token数: {total_tokens / total_samples:.2f}")
