import json
import os


def analyze_jsonl_stats(filepath):
    max_length = 0
    min_length = float('inf')
    total_length = 0
    count = 0
    count_over_512 = 0
    moss_count = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # 假设每个JSON对象中有一个'text'字段包含文本内容
            text = data.get('text', '')
            length = len(text)

            if "MOSS" in text:
                print(text)
                moss_count += 1

            if length > 512:
                count_over_512 += 1

            max_length = max(max_length, length)
            min_length = min(min_length, length)
            total_length += length
            count += 1

    if count == 0:
        return None

    avg_length = total_length / count

    return {
        'max_length': max_length,
        'min_length': min_length,
        'avg_length': avg_length,
        'total_samples': count,
        'count_over_512': count_over_512,
        'moss_count': moss_count
    }


if __name__ == "__main__":
    jsonl_file = "../dataset/pretrain_hq_v2.jsonl"
    stats = analyze_jsonl_stats(jsonl_file)

    if stats:
        print("JSONL文件统计结果:")
        print(f"最大字符长度: {stats['max_length']}")
        print(f"最小字符长度: {stats['min_length']}")
        print(f"平均字符长度: {stats['avg_length']:.2f}")
        print(f"总样本数: {stats['total_samples']}")
        print(f"超过512长度的个数: {stats['count_over_512']}")
        print(f"contain moss的个数: {stats['moss_count']}")
    else:
        print("文件为空或没有有效数据")