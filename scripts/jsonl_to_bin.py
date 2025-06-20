import json
import torch
import os
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm


def preprocess_jsonl_to_bin(jsonl_path, output_dir, tokenizer_name, max_length=512, batch_size=1000):
    """
    将JSONL文件预处理为二进制文件

    参数:
    jsonl_path: JSONL文件路径
    output_dir: 输出目录
    tokenizer_name: 分词器名称或路径
    max_length: 最大序列长度
    batch_size: 每个二进制文件包含的样本数量
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # 计算JSONL文件的行数
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    # 处理JSONL文件
    samples = []
    file_idx = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_idx, line in tqdm(enumerate(f), total=total_lines, desc="处理JSONL"):
            try:
                data = json.loads(line.strip())
                text = str(data.get('text', ''))

                # 使用分词器处理文本
                encoding = tokenizer(
                    text,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )

                input_ids = encoding.input_ids.squeeze()
                attention_mask = encoding.attention_mask.squeeze()

                # 创建样本
                sample = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                }

                samples.append(sample)

                # 当样本数量达到batch_size时，保存为一个二进制文件
                if len(samples) >= batch_size:
                    output_path = os.path.join(output_dir, f"pretrain_data_{file_idx}.pt")
                    torch.save(samples, output_path)
                    samples = []
                    file_idx += 1

            except Exception as e:
                print(f"处理第{line_idx + 1}行时出错: {e}")

    # 保存剩余的样本
    if samples:
        output_path = os.path.join(output_dir, f"pretrain_data_{file_idx}.pt")
        torch.save(samples, output_path)

    # 创建索引文件，记录总文件数
    index_path = os.path.join(output_dir, "index.json")
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump({
            "num_files": file_idx + 1,
            "samples_per_file": batch_size,
            "tokenizer": tokenizer_name,
            "max_length": max_length,
            "total_samples": total_lines
        }, f)

    print(f"预处理完成！共处理 {total_lines} 个样本，生成 {file_idx + 1} 个二进制文件")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将JSONL文件预处理为二进制文件")
    parser.add_argument("--jsonl_path", type=str, default="../dataset/pretrain_hq_v5.jsonl",  help="JSONL文件路径")
    parser.add_argument("--output_dir", type=str, default="../dataset/bin/",  help="输出目录")
    parser.add_argument("--tokenizer", type=str,  default="../model/", help="分词器名称或路径")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=500000, help="每个二进制文件包含的样本数量")

    args = parser.parse_args()

    preprocess_jsonl_to_bin(
        args.jsonl_path,
        args.output_dir,
        args.tokenizer,
        args.max_length,
        args.batch_size
    )
