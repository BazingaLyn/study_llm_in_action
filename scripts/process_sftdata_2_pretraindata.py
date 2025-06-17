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
                if len(text) > 512 or len(text) < 200 or '---' in text or "感谢你" in text or "期待您" in text:
                    continue
                text = text.replace('MOSS', 'Bazinga')  # 新增替换逻辑
                newdata['text'] = text

                json.dump(newdata, outfile, ensure_ascii=False)
                outfile.write('\n')
            if len(data.get('history', '')) != 0:
                for his in data.get('history', ''):
                    for each_his in his:
                        text = each_his + '<|endoftext|>'
                        if len(text) > 512 or len(text) < 200 or '---' in text or "感谢你" in text or "期待您" in text:
                            continue
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


if __name__ == '__main__':
    # ... 其余代码保持不变 ...
    input_file = "D:\workspace_2025\study_llm_in_action\dataset\sft_data_zh.jsonl"
    output_file = "../dataset/pretrain_hq_v3.jsonl"

    process_sftdata_2_pretrain_data(input_file, output_file)
    print(f"处理已经完成，新文件已保存至: {output_file}")