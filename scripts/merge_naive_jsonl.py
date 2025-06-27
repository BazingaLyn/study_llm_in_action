import os
from tqdm import tqdm

def merge_jsonl_files(input_files, output_file):
    """简单合并多个JSONL文件到一个输出文件，带双层进度条
    
    Args:
        input_files (list): 输入文件路径列表
        output_file (str): 输出文件路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 统计所有文件的总行数
    total_lines = 0
    file_line_counts = []
    
    print("统计文件行数...")
    for file_path in input_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
                total_lines += line_count
                file_line_counts.append((file_path, line_count))
        else:
            print(f"警告: 文件不存在 - {file_path}")
            file_line_counts.append((file_path, 0))
    
    print(f"总计 {total_lines} 行数据将被合并")
    
    # 打开输出文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # 创建文件级别的进度条
        file_progress = tqdm(file_line_counts, desc="文件进度", position=0)
        
        # 处理每个输入文件
        for file_path, line_count in file_progress:
            if line_count == 0:
                continue
                
            file_name = os.path.basename(file_path)
            file_progress.set_description(f"合并文件: {file_name}")
            
            with open(file_path, 'r', encoding='utf-8') as infile:
                # 创建行级别的进度条
                line_progress = tqdm(total=line_count, desc=f"行进度", 
                                    position=1, leave=False)
                
                # 逐行处理
                for i, line in enumerate(infile):
                    outfile.write(line)
                    line_progress.update(1)
                
                line_progress.close()
    
    print(f"\n合并完成! 所有数据已保存到: {output_file}")
    print(f"总计合并了 {total_lines} 行数据")


if __name__ == '__main__':
    input_file_1 = r"D:\workspace_2025\study_llm_in_action\dataset\sft_data_zh.jsonl"
    input_file_2 = r"D:\workspace_2025\study_llm_in_action\dataset\wiki_zh.jsonl"
    input_file_3 = r"D:\workspace_2025\study_llm_in_action\dataset\wikipedia-cn-20230720.jsonl"
    input_file_4 = r"D:\workspace_2025\study_llm_in_action\dataset\CNewSum_v2.jsonl"

    output_file = r"D:\workspace_2025\study_llm_in_action\dataset\pretrain_native_dataset.jsonl"
    
    # 合并所有输入文件
    input_files = [input_file_1, input_file_2, input_file_3, input_file_4]
    merge_jsonl_files(input_files, output_file)
