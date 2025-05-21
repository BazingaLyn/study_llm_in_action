import os
import glob
import json
import re
from tqdm import tqdm

def is_mostly_chinese(text, threshold=0.9):
    if not text or len(text) == 0:
        return False
    
    # 匹配中文字符的正则表达式
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    
    # 计算中文字符数量
    chinese_chars = chinese_pattern.findall(text)
    chinese_ratio = len(chinese_chars) / len(text)
    
    return chinese_ratio >= threshold

def read_and_process_part_files(directory="downloads", output_dir=None):
    """
    读取指定目录下所有以'part'开头的JSONL文件，
    并根据content字段长度将内容分类保存到不同文件中
    
    参数:
        directory (str): 要搜索的目录路径，默认为'downloads'
        output_dir (str): 输出文件保存目录，默认与输入目录相同
    """
    # 确保目录路径存在
    if not os.path.exists(directory):
        print(f"错误: 目录 '{directory}' 不存在")
        return {}
    
    # 设置输出目录
    if output_dir is None:
        output_dir = directory
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义输出文件路径
    output_files = {
        512: os.path.join(output_dir, "pretrain_512.jsonl"),
        1024: os.path.join(output_dir, "pretrain_1024.jsonl"),
        2048: os.path.join(output_dir, "pretrain_2048.jsonl"),
        4096: os.path.join(output_dir, "pretrain_4096.jsonl")
    }

    for length, file_path in output_files.items():
        if not os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf-8') as f:
                pass  # 创建空文件
            print(f"已创建输出文件: {os.path.basename(file_path)}")
    
    # 初始化计数器
    counters = {512: 0, 1024: 0, 2048: 0, 4096: 0}
    filtered_non_chinese = 0
    
    # 获取所有以'part'开头的文件路径
    pattern = os.path.join(directory, "part*")
    part_files = glob.glob(pattern)
    
    if not part_files:
        print(f"在 '{directory}' 目录中没有找到以'part'开头的文件")
        return {}
    
    print(f"找到 {len(part_files)} 个以'part'开头的文件，开始处理...")
    
    # 使用tqdm创建文件处理进度条
    for file_path in tqdm(part_files, desc="处理文件进度", unit="文件"):
        file_name = os.path.basename(file_path)
        
        try:
            # 获取文件行数以设置进度条
            line_count = 0
            with open(file_path, 'r', encoding='utf-8') as f:
                for _ in f:
                    line_count += 1
            
            # 逐行读取JSONL文件
            with open(file_path, 'r', encoding='utf-8') as f:
                # 使用tqdm创建行处理进度条
                for line in tqdm(f, total=line_count, desc=f"处理 {file_name}", unit="行", leave=False):
                    try:
                        # 解析JSON
                        data = json.loads(line.strip())
                        
                        # 检查是否包含content字段
                        if 'content' in data and isinstance(data['content'], str):
                            content = data['content']
                            
                            # 判断内容是否主要为中文
                            if not is_mostly_chinese(content):
                                filtered_non_chinese += 1
                                # 每10条非中文内容保留1条
                                if filtered_non_chinese % 20 != 0:
                                    continue
                            
                            content_length = len(content)
                            
                            # 根据content长度分类
                            if content_length < 512:
                                output_file = output_files[512]
                                counters[512] += 1
                            elif content_length < 1024:
                                output_file = output_files[1024]
                                counters[1024] += 1
                            elif content_length < 2048:
                                output_file = output_files[2048]
                                counters[2048] += 1
                            else:
                                output_file = output_files[4096]
                                counters[4096] += 1
                            
                            # 追加到对应文件
                            with open(output_file, 'a', encoding='utf-8') as out_f:
                                out_f.write(line)
                        else:
                            tqdm.write(f"  警告: 缺少content字段或格式不正确，已跳过")
                    
                    except json.JSONDecodeError:
                        tqdm.write(f"  警告: 不是有效的JSON格式，已跳过")
            
        except Exception as e:
            tqdm.write(f"处理文件 '{file_name}' 时出错: {str(e)}")
            # 尝试使用其他编码
            try:
                # 获取文件行数以设置进度条
                line_count = 0
                with open(file_path, 'r', encoding='gbk') as f:
                    for _ in f:
                        line_count += 1
                
                # 逐行读取JSONL文件（GBK编码）
                with open(file_path, 'r', encoding='gbk') as f:
                    # 使用tqdm创建行处理进度条
                    for line in tqdm(f, total=line_count, desc=f"处理 {file_name} (GBK)", unit="行", leave=False):
                        try:
                            # 解析JSON
                            data = json.loads(line.strip())
                            
                            # 检查是否包含content字段
                            if 'content' in data and isinstance(data['content'], str):
                                content = data['content']
                                
                                # 判断内容是否主要为中文
                                if not is_mostly_chinese(content):
                                    filtered_non_chinese += 1
                                    continue
                                
                                content_length = len(content)
                                
                                # 根据content长度分类
                                if content_length < 512:
                                    output_file = output_files[512]
                                    counters[512] += 1
                                elif content_length < 1024:
                                    output_file = output_files[1024]
                                    counters[1024] += 1
                                elif content_length < 2048:
                                    output_file = output_files[2048]
                                    counters[2048] += 1
                                else:
                                    output_file = output_files[4096]
                                    counters[4096] += 1
                                
                                # 追加到对应文件
                                with open(output_file, 'a', encoding='utf-8') as out_f:
                                    out_f.write(line)
                            else:
                                tqdm.write(f"  警告: 缺少content字段或格式不正确，已跳过")
                        
                        except json.JSONDecodeError:
                            tqdm.write(f"  警告: 不是有效的JSON格式，已跳过")
                
                tqdm.write(f"  已使用GBK编码完成文件 {file_name} 的处理")
                
            except Exception as e2:
                tqdm.write(f"使用GBK编码处理文件 '{file_name}' 也失败: {str(e2)}")
    
    # 打印统计信息
    print("\n处理完成! 统计信息:")
    print(f"过滤的非中文内容: {filtered_non_chinese} 条")
    print(f"长度 < 512: {counters[512]} 条记录，保存在 {os.path.basename(output_files[512])}")
    print(f"长度 < 1024: {counters[1024]} 条记录，保存在 {os.path.basename(output_files[1024])}")
    print(f"长度 < 2048: {counters[2048]} 条记录，保存在 {os.path.basename(output_files[2048])}")
    print(f"长度 >= 2048: {counters[4096]} 条记录，保存在 {os.path.basename(output_files[4096])}")
    
    return counters

def main():
    """主函数"""
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建downloads目录的完整路径
    downloads_dir = os.path.join(os.path.dirname(current_dir), "downloads")
    # 构建输出目录的完整路径
    output_dir = os.path.join(os.path.dirname(current_dir), "dataset")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"正在搜索目录: {downloads_dir}")
    print(f"输出目录: {output_dir}")
    
    # 读取并处理文件
    read_and_process_part_files(downloads_dir, output_dir)

if __name__ == "__main__":
    main()