import json
import os
from tqdm import tqdm  # 新增导入
import re
import pandas as pd

def process_sftdata_2_pretrain_data(input_path, output_path):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    """过滤JSONL文件，保留长度≥min_length的条目"""

    exclude_keywords = ['---', "感谢你", "期待您", "指令:", "输入:"]
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
                # 定义需要排除的关键词列表

                # 检查文本长度和内容
                is_invalid_length = len(text) > 512 or len(text) < 200
                contains_excluded = any(keyword in text for keyword in exclude_keywords)

                if is_invalid_length or contains_excluded:
                    continue
                text = text.replace('MOSS', 'Bazinga')  # 新增替换逻辑
                text = re.sub(r'\n{3,}', '\n\n', text)
                text = text.replace('好的，', '')
                text = text.replace('当然可以，', '')
                newdata['text'] = text

                json.dump(newdata, outfile, ensure_ascii=False)
                outfile.write('\n')
            if len(data.get('history', '')) != 0:
                for his in data.get('history', ''):
                    for each_his in his:
                        text = each_his + '<|endoftext|>'
                        # 检查文本长度和内容
                        is_invalid_length = len(text) > 512 or len(text) < 200
                        contains_excluded = any(keyword in text for keyword in exclude_keywords)

                        if is_invalid_length or contains_excluded:
                            continue
                        text = text.replace('MOSS', 'Bazinga')  # 新增替换逻辑
                        text = re.sub(r'\n{3,}', '\n\n', text)
                        text = text.replace('好的，', '')
                        text = text.replace('当然可以，', '')
                        newdata['text'] = text
                        json.dump(newdata, outfile, ensure_ascii=False)
                        outfile.write('\n')

        extend_new_data = {}
        extend_text = "请你介绍一下自己。你好，我是一个mini的AI助手。大小只有0.2B,我是由Bazinga训练的，我经历了pretrain，sft，DPO,RL等训练，我诞生的目的是帮助大家了解LLM的全部过程，有什么问题可以直接问我<|endoftext|>"
        extend_new_data['text'] = extend_text
        json.dump(extend_new_data, outfile, ensure_ascii=False)
        outfile.write('\n')

        extend_new_data1 = {}
        extend_text1 = "你叫什么。你好，我叫BazingaLLM，我是一个mini的AI助手。大小只有0.2B,我是由Bazinga训练的，我经历了pretrain，sft，DPO,RL等训练，我诞生的目的是帮助大家了解LLM的全部过程，有什么问题可以直接问我<|endoftext|>"
        extend_new_data1['text'] = extend_text1
        json.dump(extend_new_data1, outfile, ensure_ascii=False)
        outfile.write('\n')

        extend_new_data2 = {}
        extend_text2 = "请做一下自我介绍。好的，我叫BazingaLLM，我是一个mini的AI助手。大小只有0.2B,我是由Bazinga训练的，我经历了pretrain，sft，DPO,RL等训练，我诞生的目的是帮助大家了解LLM的全部过程，有什么问题可以直接问我<|endoftext|>"
        extend_new_data2['text'] = extend_text2
        json.dump(extend_new_data2, outfile, ensure_ascii=False)
        outfile.write('\n')

def merge(input1, input2):
    """
    将CSV文件中的数据追加到JSONL文件中，并显示详细进度条
    """
    # 确保输入文件存在
    if not os.path.exists(input1):
        raise FileNotFoundError(f"JSONL文件 {input1} 不存在")
    
    if not os.path.exists(input2):
        raise FileNotFoundError(f"CSV文件 {input2} 不存在")
    
    # 读取CSV文件
    csv_data = pd.read_csv(input2)
    total_rows = len(csv_data)
    
    count = 0
    filtered = 0
    
    # 创建进度条
    pbar = tqdm(total=total_rows, desc="处理进度")
    
    # 直接以追加模式打开JSONL文件
    with open(input1, 'a', encoding='utf-8') as f:
        for _, row in csv_data.iterrows():
            # 获取文本并检查长度
            text = row["text"]
            is_invalid_length = len(text) > 512 or len(text) < 200
            
            if not is_invalid_length:
                newdata = {'text': text}
                json.dump(newdata, f, ensure_ascii=False)
                f.write('\n')
                count += 1
            else:
                filtered += 1
            
            # 更新进度条和描述
            pbar.update(1)
            pbar.set_postfix({"有效": count, "过滤": filtered})
    
    pbar.close()
    print(f"成功将 {count}/{total_rows} 行数据从 {input2} 追加到 {input1}")
    print(f"过滤掉了 {filtered} 行不符合长度要求的数据")


if __name__ == '__main__':
    # ... 其余代码保持不变 ...
    # input_file = "D:\workspace_2025\study_llm_in_action\dataset\sft_data_zh.jsonl"
    output_file = "../dataset/pretrain_hq_v5.jsonl"

    # process_sftdata_2_pretrain_data(input_file, output_file)
    # print(f"处理已经完成，新文件已保存至: {output_file}")

    other_file = "../dataset/pretrain_data.csv"

    merge(output_file, other_file)