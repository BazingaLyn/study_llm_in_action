import json
import os
import re
from tqdm import tqdm

def fix_jsonl_file(input_file, output_file, buffer_size=1024*1024):
    """
    修复JSONL文件中的换行问题
    使用流式处理方式处理大文件，避免内存溢出
    """
    # 获取文件大小用于进度显示
    file_size = os.path.getsize(input_file)
    processed_size = 0
    
    # 用于存储不完整的行
    incomplete_line = ""
    in_json_object = False
    current_object = ""
    brace_count = 0
    
    with open(input_file, 'r', encoding='utf-8', errors='replace') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile, \
         tqdm(total=file_size, unit='B', unit_scale=True, desc="处理进度") as pbar:
        
        while True:
            # 读取一块数据
            chunk = infile.read(buffer_size)
            if not chunk:
                break
                
            processed_size += len(chunk.encode('utf-8'))
            pbar.update(len(chunk.encode('utf-8')))
            
            # 将不完整的行与当前块合并
            data = incomplete_line + chunk
            lines = data.split('\n')
            
            # 最后一行可能不完整，保存到下一次迭代
            incomplete_line = lines[-1]
            
            # 处理完整的行
            for i, line in enumerate(lines[:-1]):
                if not in_json_object:
                    # 检查是否是新的JSON对象开始
                    if line.strip().startswith('{'):
                        in_json_object = True
                        current_object = line
                        brace_count = line.count('{') - line.count('}')
                        
                        # 如果在同一行结束，直接写入
                        if brace_count == 0 and line.strip().endswith('}'):
                            try:
                                # 验证JSON格式
                                json_obj = json.loads(line)
                                outfile.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
                                in_json_object = False
                                current_object = ""
                            except json.JSONDecodeError:
                                # 如果解析失败，可能是多行对象
                                pass
                else:
                    # 继续累积当前对象
                    current_object += "\n" + line
                    brace_count += line.count('{') - line.count('}')
                    
                    # 检查对象是否结束
                    if brace_count == 0 and line.strip().endswith('}'):
                        try:
                            # 尝试解析并修复JSON对象
                            json_obj = json.loads(current_object)
                            
                            # 特别处理text字段，移除内部换行符
                            if 'text' in json_obj and isinstance(json_obj['text'], str):
                                # 将text字段中的换行符替换为空格
                                json_obj['text'] = re.sub(r'\s+', ' ', json_obj['text'])
                            
                            # 写入修复后的对象
                            outfile.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
                            
                            in_json_object = False
                            current_object = ""
                        except json.JSONDecodeError as e:
                            # 如果仍然无法解析，可能是对象尚未结束
                            # 或者有其他格式问题
                            if i == len(lines) - 2:  # 如果是块中的最后一行
                                pass  # 继续到下一个块
                            else:
                                # 尝试修复明显的格式问题
                                # 例如，将text字段中的换行符替换为空格
                                current_object = re.sub(r'"text"\s*:\s*"([^"]*?)\n([^"]*?)"', 
                                                       r'"text": "\1 \2"', 
                                                       current_object)
        
        # 处理最后一个不完整的行
        if incomplete_line:
            if in_json_object:
                current_object += "\n" + incomplete_line
            else:
                current_object = incomplete_line
                
            try:
                json_obj = json.loads(current_object)
                
                # 特别处理text字段
                if 'text' in json_obj and isinstance(json_obj['text'], str):
                    json_obj['text'] = re.sub(r'\s+', ' ', json_obj['text'])
                
                outfile.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
            except json.JSONDecodeError:
                print(f"警告：无法解析最后一个对象: {current_object[:100]}...")

def main():
    input_file = "../dataset/pretrain_large_new.jsonl"
    output_file = "../dataset/pretrain_large_fixed.jsonl"
    
    print(f"开始修复文件: {input_file}")
    print(f"输出文件: {output_file}")
    
    fix_jsonl_file(input_file, output_file)
    
    print("处理完成！")
    print(f"修复后的文件已保存到: {output_file}")

if __name__ == "__main__":
    main()