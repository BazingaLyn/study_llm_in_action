import json
import torch
import os
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
from dataset.lm_dataset import PretrainDataset, BinaryPretrainDataset


# 测试两个数据集是否一致的函数
def test_dataset_consistency(jsonl_path, bin_dir, tokenizer_name, num_samples=100):
    """测试原始数据集和二进制数据集的输出是否一致"""
    print(f"加载分词器: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


    print("创建原始数据集...")
    original_dataset = PretrainDataset(jsonl_path, tokenizer)

    print("创建二进制数据集...")
    binary_dataset = BinaryPretrainDataset(bin_dir, tokenizer)

    print(f"原始数据集大小: {len(original_dataset)}")
    print(f"二进制数据集大小: {len(binary_dataset)}")

    # 确保测试样本数不超过数据集大小
    num_samples = min(num_samples, len(original_dataset), len(binary_dataset))

    print(f"\n开始测试 {num_samples} 个样本的一致性...")

    # 随机选择索引进行比较
    indices = np.random.choice(min(len(original_dataset), len(binary_dataset)),
                               size=num_samples, replace=False)

    all_consistent = True

    for i, idx in enumerate(indices):
        print(f"\n测试样本 {i + 1}/{num_samples} (索引 {idx}):")

        # 获取原始数据集的样本
        orig_X, orig_Y, orig_mask = original_dataset[idx]

        # 获取二进制数据集的样本
        bin_X, bin_Y, bin_mask = binary_dataset[idx]

        # 检查形状是否一致
        shape_consistent = (orig_X.shape == bin_X.shape and
                            orig_Y.shape == bin_Y.shape and
                            orig_mask.shape == bin_mask.shape)

        # 检查内容是否一致
        content_consistent = (torch.all(orig_X == bin_X) and
                              torch.all(orig_Y == bin_Y) and
                              torch.all(orig_mask == bin_mask))

        if shape_consistent and content_consistent:
            print(f"✓ 样本一致")
        else:
            all_consistent = False
            print(f"✗ 样本不一致")

            if not shape_consistent:
                print(f"  形状不一致:")
                print(f"  - 原始: X={orig_X.shape}, Y={orig_Y.shape}, mask={orig_mask.shape}")
                print(f"  - 二进制: X={bin_X.shape}, Y={bin_Y.shape}, mask={bin_mask.shape}")

            if not content_consistent:
                if not torch.all(orig_X == bin_X):
                    print(f"  X 不一致")
                if not torch.all(orig_Y == bin_Y):
                    print(f"  Y 不一致")
                if not torch.all(orig_mask == bin_mask):
                    print(f"  mask 不一致")

    print("\n测试结果:")
    if all_consistent:
        print("✓ 所有测试样本都一致！两个数据集实现完全相同。")
    else:
        print("✗ 存在不一致的样本，请检查实现。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试数据集一致性")
    parser.add_argument("--jsonl_path", type=str, default="../dataset/pretrain_hq_v5.jsonl", help="JSONL文件路径")
    parser.add_argument("--bin_dir", type=str, default="../dataset/bin/", help="二进制文件目录")
    parser.add_argument("--tokenizer", type=str, default="../model/", help="分词器名称")
    parser.add_argument("--num_samples", type=int, default=10, help="测试样本数")

    args = parser.parse_args()

    test_dataset_consistency(
        args.jsonl_path,
        args.bin_dir,
        args.tokenizer,
        args.num_samples
    )