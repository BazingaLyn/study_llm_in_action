import random
import json
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
import os
import gc
import logging
from tqdm import tqdm
import multiprocessing as mp
from itertools import islice

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

random.seed(42)


def train_tokenizer():
    # 读取JSONL文件并提取文本数据，使用批处理方式
    def read_texts_from_jsonl_batched(file_path, batch_size=100000):
        batch = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        # 只保留文本，减少内存占用
                        text = data.get('text', '')
                        if text:
                            batch.append(text)
                            if len(batch) >= batch_size:
                                yield batch
                                batch = []
                                # 强制垃圾回收
                                gc.collect()
                    except json.JSONDecodeError:
                        logger.warning(f"跳过无效的JSON行")
                        continue
                    except Exception as e:
                        logger.warning(f"处理行时出错: {str(e)}")
                        continue
                
                # 返回最后一批
                if batch:
                    yield batch
        except Exception as e:
            logger.error(f"读取文件 {file_path} 时出错: {str(e)}")

    # 分片处理大文件
    def process_file_in_chunks(file_path, output_dir, chunk_size=2000000, max_chunks=None):
        """将大文件分割成多个小文件进行处理"""
        os.makedirs(output_dir, exist_ok=True)
        chunk_files = []
        
        chunk_num = 0
        processed_lines = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            while True:
                if max_chunks and chunk_num >= max_chunks:
                    break
                
                # 读取一个块的数据
                chunk = list(islice(f, chunk_size))
                if not chunk:
                    break
                
                chunk_file = os.path.join(output_dir, f"chunk_{chunk_num}.jsonl")
                with open(chunk_file, 'w', encoding='utf-8') as chunk_f:
                    for line in chunk:
                        try:
                            # 验证JSON格式
                            data = json.loads(line)
                            if 'text' in data:
                                chunk_f.write(line)
                        except:
                            # 跳过无效行
                            continue
                
                chunk_files.append(chunk_file)
                chunk_num += 1
                processed_lines += len(chunk)
                logger.info(f"已处理 {processed_lines} 行，创建分片文件: {chunk_file}")
                
                # 强制垃圾回收
                gc.collect()
        
        return chunk_files

    data_path = '../dataset/pretrain_large_new.jsonl'
    
    # 创建临时目录存放分片文件
    temp_dir = "../temp_chunks"
    
    # 分片处理大文件
    logger.info("开始分片处理大文件...")
    chunk_files = process_file_in_chunks(data_path, temp_dir)
    logger.info(f"文件分片完成，共创建 {len(chunk_files)} 个分片文件")

    # 初始化tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 定义特殊token
    special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]

    # 设置训练器并添加特殊token
    trainer = trainers.BpeTrainer(
        vocab_size=7200,
        special_tokens=special_tokens,  # 确保这三个token被包含
        show_progress=True,
        # min_frequency=5,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # 分批训练tokenizer
    logger.info("开始训练tokenizer...")
    for i, chunk_file in enumerate(chunk_files):
        logger.info(f"处理分片 {i+1}/{len(chunk_files)}: {chunk_file}")
        
        # 分批读取文本数据
        for batch_idx, batch in enumerate(read_texts_from_jsonl_batched(chunk_file)):
            logger.info(f"  训练批次 {batch_idx+1}，样本数: {len(batch)}")
            
            # 训练tokenizer (增量训练)
            tokenizer.train_from_iterator(batch, trainer=trainer)
            
            # 清理内存
            del batch
            gc.collect()
    
    # 设置解码器
    tokenizer.decoder = decoders.ByteLevel()

    # 检查特殊token的索引
    try:
        assert tokenizer.token_to_id("<|endoftext|>") == 0
        assert tokenizer.token_to_id("<|im_start|>") == 1
        assert tokenizer.token_to_id("<|im_end|>") == 2
        logger.info("特殊token索引检查通过")
    except AssertionError:
        logger.warning("特殊token索引检查失败，可能需要手动调整")

    # 保存tokenizer
    tokenizer_dir = "../model/7200_tokenizer/"
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save("../model/7200_tokenizer/")
    logger.info(f"Tokenizer已保存到 {tokenizer_dir}")

    # 手动创建配置文件
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": {
            "0": {
                "content": "<|endoftext|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "<|im_start|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "<|im_end|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "legacy": True,
        "model_max_length": 32768,
        "pad_token": "<|endoftext|>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<|endoftext|>",
        "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{{ '<|im_start|>system\\n' + system_message + '<|im_end|>\\n' }}{% else %}{{ '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\\n' + content + '<|im_end|>\\n<|im_start|>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<|im_end|>' + '\\n' }}{% endif %}{% endfor %}"
    }

    # 保存配置文件
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)

    # 清理临时文件
    logger.info("清理临时文件...")
    for chunk_file in chunk_files:
        try:
            os.remove(chunk_file)
        except:
            pass
    try:
        os.rmdir(temp_dir)
    except:
        pass

    logger.info("Tokenizer训练完成并保存。")


def eval_tokenizer():
    from transformers import AutoTokenizer

    # 加载预训练的tokenizer
    tokenizer = AutoTokenizer.from_pretrained("../model/")

    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
    ]
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    print(new_prompt)

    # 获取实际词汇表长度（包括特殊符号）
    actual_vocab_size = len(tokenizer)
    print('tokenizer实际词表长度：', actual_vocab_size)

    model_inputs = tokenizer(new_prompt)
    print('encoder长度：', len(model_inputs['input_ids']))

    input_ids = model_inputs['input_ids']
    response = tokenizer.decode(input_ids, skip_special_tokens=False)
    print('decoder和原始文本是否一致：', response == new_prompt)


def main():
    # # 限制内存使用
    # import resource
    # # 软限制为16GB (仅在Linux/Unix系统上有效)
    # if os.name != 'nt':  # 不是Windows
    #     resource.setrlimit(resource.RLIMIT_AS, (16 * 1024 * 1024 * 1024, -1))
    
    # # 设置进程数量限制
    # mp.set_start_method('spawn', force=True)
    
    train_tokenizer()
    eval_tokenizer()


if __name__ == '__main__':
    main()