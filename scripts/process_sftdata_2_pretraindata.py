import json
import os
from tqdm import tqdm  # 新增导入


def process_sftdata_2_pretrain_data(input_path, output_path):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    """过滤JSONL文件，保留长度≥min_length的条目"""

    # 获取文件总行数用于进度条
    with open(input_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    with open(input_path, 'r', encoding='utf-8') as infile, \
            open(output_path, 'w', encoding='utf-8') as outfile:

        for line in tqdm(infile, total=total_lines, desc="处理进度"):  # 添加tqdm包装
            newdata = {}
            data = json.loads(line)
            # ... 其余代码保持不变 ...
            input = data.get('input', '')
            output = data.get('output', '')
            if len(data.get('history', '')) == 0 and input != '' and output != '':
                text = input + output + '<|endoftext|>'
                if len(text) > 512 or len(text) < 200 or '---' in text:
                    continue
                newdata['text'] = text
                json.dump(newdata, outfile, ensure_ascii=False)
                outfile.write('\n')
            if len(data.get('history', '')) != 0:
                for his in data.get('history', ''):
                    for each_his in his:
                        text = each_his + '<|endoftext|>'
                        if len(text) > 512 or len(text) < 200 or '---' in text:
                            continue
                        newdata['text'] = text
                        json.dump(newdata, outfile, ensure_ascii=False)
                        outfile.write('\n')


if __name__ == '__main__':
    # ... 其余代码保持不变 ...
    input_file = "../dataset/sft_data_zh.jsonl"
    output_file = "../dataset/pretrain_hq_v2.jsonl"

    process_sftdata_2_pretrain_data(input_file, output_file)
    print(f"处理已经完成，新文件已保存至: {output_file}")