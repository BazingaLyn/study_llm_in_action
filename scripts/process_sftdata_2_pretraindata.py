import json
import os
from tqdm import tqdm  # 新增导入
import re
import pandas as pd

def process_sftdata_2_pretrain_data(input_path, output_path):
    """将SFT数据处理为预训练数据格式"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 获取文件总行数用于进度条
    with open(input_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    with open(input_path, 'r', encoding='utf-8') as infile, \
            open(output_path, 'w', encoding='utf-8') as outfile:

        for line in tqdm(infile, total=total_lines, desc="处理进度"):
            process_data_line(line, outfile)
            
        # 添加扩展数据
        add_extension_data(outfile)


def should_exclude_text(text, min_length=200, max_length=512):
    """检查文本是否应该被排除
    
    Args:
        text (str): 要检查的文本
        min_length (int): 最小文本长度
        max_length (int): 最大文本长度
        
    Returns:
        bool: 如果应该排除则返回True，否则返回False
    """
    # 首次预训练阶段需要排除在sft中出现的文本
    # 首次预训练阶段也需要排除编程相关的相对比较复杂的文本
    exclude_keywords = [
        '---','____', "感谢你", "期待您", "指令:", "输入:",
        "【文章内容】", "【结束】", "轮对话】", 
        "[文章](", "[您需要的", "请继续撰写", "请稍等片刻", "手机号码", "电话号码", "内容省略", "Python", "python", "代码", "代码块", "```",
        "java","Java","JAVA","C++","c++","C","c","C#","c#","C#","JavaScript","javascript","JavaScript","TypeScript","typescript","TypeScript","给你两个角色信息如下","角色信息如下","完成一段对话",
    ]
    
    # 检查文本长度
    if len(text) > max_length or len(text) < min_length:
        return True
        
    # 检查是否包含排除关键词
    if any(keyword in text for keyword in exclude_keywords):
        return True
        
    return False


def clean_text(text):
    """清洗文本
    
    Args:
        text (str): 要清洗的文本
        
    Returns:
        str: 清洗后的文本
    """
    # 替换特定字符串
    text = text.replace('MOSS', 'Bazinga')
    
    # 处理多余的换行符
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 移除特定开头语
    text = text.replace('好的，', '')
    text = text.replace('当然可以，', '')
    
    # 去除开头的换行符
    if text.startswith('\n'):
        text = text.lstrip('\n')

    if text.startswith('当然，'):
        text = text.lstrip('当然，')

    if text.startswith('当然可以！'):
        text = text.lstrip('当然可以！')
    return text


def process_data_line(line, outfile):
    """处理单行数据
    
    Args:
        line (str): JSON行数据
        outfile (file): 输出文件对象
    """
    try:
        data = json.loads(line)
        
        # 处理没有历史记录的情况
        input_text = data.get('input', '')
        output_text = data.get('output', '')

        text = data.get('text', '')
        if len(text) != 0:
            process_single_text(text, outfile)

        if len(data.get('history', '')) == 0 and input_text != '' and output_text != '':
            process_single_text(input_text + output_text, outfile)
        
        # 处理有历史记录的情况
        if len(data.get('history', '')) != 0:
            for his in data.get('history', ''):
                for each_his in his:
                    process_single_text(each_his, outfile)
    except json.JSONDecodeError:
        print(f"警告: 无法解析JSON行: {line[:50]}...")


def process_single_text(text, outfile):
    """处理单条文本
    
    Args:
        text (str): 要处理的文本
        outfile (file): 输出文件对象
    """
    text = text + '<|endoftext|>'
    
    # 检查是否应该排除
    if should_exclude_text(text):
        return
    
    # 清洗文本
    text = clean_text(text)
    
    # 写入输出文件
    newdata = {'text': text}
    json.dump(newdata, outfile, ensure_ascii=False)
    outfile.write('\n')


def add_extension_data(outfile):
    """添加扩展数据
    
    Args:
        outfile (file): 输出文件对象
    """
    extension_texts = [
        "请你介绍一下自己。你好，我是一个mini的AI助手。大小只有0.2B,我是由Bazinga训练的，我经历了pretrain，sft，DPO,RL等训练，我诞生的目的是帮助大家了解LLM的全部过程，有什么问题可以直接问我<|endoftext|>",
        "你叫什么。你好，我叫BazingaLLM，我是一个mini的AI助手。大小只有0.2B,我是由Bazinga训练的，我经历了pretrain，sft，DPO,RL等训练，我诞生的目的是帮助大家了解LLM的全部过程，有什么问题可以直接问我<|endoftext|>",
        "请做一下自我介绍。好的，我叫BazingaLLM，我是一个mini的AI助手。大小只有0.2B,我是由Bazinga训练的，我经历了pretrain，sft，DPO,RL等训练，我诞生的目的是帮助大家了解LLM的全部过程，有什么问题可以直接问我<|endoftext|>"
    ]
    
    for ext_text in extension_texts:
        newdata = {'text': ext_text}
        json.dump(newdata, outfile, ensure_ascii=False)
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
    input_file = "D:\workspace_2025\study_llm_in_action\dataset\pretrain_native_dataset.jsonl"
    output_file = "../dataset/pretrain_hq_v7.jsonl"

    process_sftdata_2_pretrain_data(input_file, output_file)
    print(f"处理已经完成，新文件已保存至: {output_file}")

    # other_file = "../dataset/pretrain_data.csv"
    #
    # merge(output_file, other_file)