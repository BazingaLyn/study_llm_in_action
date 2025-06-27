
import json

def extract_json_with_interval(input_file, output_file, interval=4):
    """
    从jsonl文件中每隔指定间隔抽取一个JSON对象，并保存到新文件
    
    参数:
        input_file (str): 输入的jsonl文件路径
        output_file (str): 输出的jsonl文件路径
        interval (int): 抽取间隔，默认为4
    """
    extracted_data = []
    
    # 读取原始jsonl文件
    with open(input_file, 'r', encoding='utf-8') as f:
        count = 0
        for line in f:
            if count % interval == 0:  # 每隔interval个对象抽取一个
                try:
                    json_obj = json.loads(line.strip())
                    extracted_data.append(json_obj)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse JSON at line {count+1}")
            count += 1
    
    # 写入新的jsonl文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in extracted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"抽取完成! 共抽取 {len(extracted_data)} 个JSON对象，保存到 {output_file}")



if __name__ == '__main__':
    # ... 其余代码保持不变 ...
    input_file = "../dataset/pretrain_hq_v7.jsonl"

    output_file = "../dataset/train_tokenizer.jsonl"
    extract_json_with_interval(input_file, output_file, interval=4)
