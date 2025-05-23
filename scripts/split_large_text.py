import json
import os
import gc
from tqdm import tqdm


def split_long_texts(input_file, output_file, max_length=1024):
    """
    处理JSONL文件，将过长的文本按照换行符切割成多个较短的文本

    参数:
    input_file: 输入JSONL文件路径
    output_file: 输出JSONL文件路径
    max_length: 文本的最大长度阈值，默认为2048
    """
    # 计算文件总行数（用于进度显示）
    print("计算文件总行数...")
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    print(f"文件总行数: {total_lines}")

    # 统计信息
    processed_lines = 0
    split_texts = 0
    output_lines = 0

    # 处理文件
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        print("开始处理文件...")
        for line in tqdm(f_in, total=total_lines, desc="处理进度"):
            try:
                # 解析JSON
                data = json.loads(line.strip())
                processed_lines += 1

                # 检查是否有text字段
                if "text" not in data:
                    # 如果没有text字段但有content字段，则转换
                    if "content" in data:
                        data["text"] = data.pop("content")
                    else:
                        # 如果既没有text也没有content，则跳过
                        continue

                text = data["text"]

                # 如果文本长度超过阈值，则进行切割
                if len(text) > max_length:
                    # 按换行符切割
                    segments = text.split("\n")

                    # 合并短段落，确保每段接近但不超过max_length
                    current_segment = ""

                    for segment in segments:
                        # 如果当前段落加上新段落不超过max_length，则合并
                        if len(current_segment) + len(segment) + 1 <= max_length:  # +1是为了算上换行符
                            if current_segment:
                                current_segment += "\n" + segment
                            else:
                                current_segment = segment
                        else:
                            # 如果当前段落已经有内容，则输出
                            if current_segment:
                                new_data = data.copy()
                                new_data["text"] = current_segment
                                f_out.write(json.dumps(new_data, ensure_ascii=False) + "\n")
                                output_lines += 1

                            # 开始新的段落
                            current_segment = segment

                            # 如果单个segment超过max_length，则进一步切割
                            while len(current_segment) > max_length:
                                # 尝试在max_length位置附近找到一个合适的切割点（如空格）
                                cut_point = max_length
                                while cut_point > max_length // 2 and current_segment[cut_point] not in [' ', '，', '。',
                                                                                                         '；', '：', '!',
                                                                                                         '?', '！', '？',
                                                                                                         ',', '.', ';',
                                                                                                         ':', '\t']:
                                    cut_point -= 1

                                # 如果找不到合适的切割点，就在max_length处强制切割
                                if cut_point <= max_length // 2:
                                    cut_point = max_length

                                # 输出切割后的前半部分
                                new_data = data.copy()
                                new_data["text"] = current_segment[:cut_point]
                                f_out.write(json.dumps(new_data, ensure_ascii=False) + "\n")
                                output_lines += 1

                                # 保留后半部分继续处理
                                current_segment = current_segment[cut_point:]

                    # 处理最后一个段落
                    if current_segment:
                        new_data = data.copy()
                        new_data["text"] = current_segment
                        f_out.write(json.dumps(new_data, ensure_ascii=False) + "\n")
                        output_lines += 1

                    split_texts += 1
                else:
                    # 如果文本长度不超过阈值，直接写入
                    f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                    output_lines += 1

                # 每处理1000行释放一次内存
                if processed_lines % 1000 == 0:
                    gc.collect()

            except json.JSONDecodeError:
                print(f"警告：跳过无效的 JSON 行")
                continue

    # 输出统计信息
    print("\n处理完成！")
    print(f"处理的总行数: {processed_lines}")
    print(f"被切割的文本数: {split_texts}")
    print(f"输出的总行数: {output_lines}")
    print(f"输出文件: {output_file}")


if __name__ == "__main__":
    # 设置输入和输出文件路径
    input_file = os.path.join('..', 'dataset', 'pretrain_large.jsonl')
    output_file = os.path.join('..', 'dataset', 'pretrain_large_new.jsonl')

    # 执行文本切割
    split_long_texts(input_file, output_file)