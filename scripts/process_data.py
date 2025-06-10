# import pandas as pd
# import json
# import os
#
#
# def csv_to_jsonl(csv_path, jsonl_path):
#     # 读取CSV文件
#     df = pd.read_csv(csv_path)
#
#     # 确保输出目录存在
#     os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
#
#     # 将DataFrame转换为JSONL格式
#     with open(jsonl_path, 'w', encoding='utf-8') as f:
#         for _, row in df.iterrows():
#             # 将每行数据转换为字典并写入文件
#             json.dump(row.to_dict(), f, ensure_ascii=False)
#             f.write('\n')
#
#
# if __name__ == "__main__":
#     # 定义输入输出路径
#     input_csv = "../dataset/pretrain_data.csv"
#     output_jsonl = "../dataset/pretrain_data.jsonl"
#
#     # 执行转换
#     csv_to_jsonl(input_csv, output_jsonl)
#     print(f"转换完成，JSONL文件已保存至: {output_jsonl}")

import json
import os


# def filter_jsonl_by_length(input_path, output_path, min_length=250):
#     """过滤JSONL文件，保留长度≥min_length的条目"""
#     with open(input_path, 'r', encoding='utf-8') as infile, \
#             open(output_path, 'w', encoding='utf-8') as outfile:
#
#         for line in infile:
#             data = json.loads(line)
#             text =  data.get('text', '') +'<|endoftext|>'
#
#             data['text'] = text
#
#             if len(text) >= min_length and len(text) <= 512:
#                 json.dump(data, outfile, ensure_ascii=False)
#                 outfile.write('\n')
#
#
# if __name__ == "__main__":
#     input_file = "../dataset/pretrain_data.jsonl"
#     output_file = "../dataset/pretrain_hq_v10.jsonl"
#
#     filter_jsonl_by_length(input_file, output_file, 250)
#     print(f"过滤完成，新文件已保存至: {output_file}")
#
# import json


def merge_jsonl_files(file1_path, file2_path, output_path):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # 以追加模式打开输出文件
    with open(output_path, 'w', encoding='utf-8') as out_file:
        # 逐行处理第一个文件
        with open(file1_path, 'r', encoding='utf-8') as f1:
            for line in f1:
                out_file.write(line)  # 直接写入行，无需解析

        # 逐行处理第二个文件
        with open(file2_path, 'r', encoding='utf-8') as f2:
            for line in f2:
                out_file.write(line)  # 直接写入行，无需解析


if __name__ == '__main__':
    # 输入文件路径
    file1 = '../dataset/wiki_zh.jsonl'
    file2 = '../dataset/wikipedia-cn-20230720.jsonl'
    # 输出文件路径
    output_file = '../dataset/pretrain_data_v1.jsonl'

    merge_jsonl_files(file1, file2, output_file)
    print(f'合并完成，结果已保存到 {output_file}')