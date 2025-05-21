### 模型结构
TODO



### 模型预训练

step1: 下载预训练的数据

默认情况下，是使用modelscope从魔搭社区社区上下载训练的语料，所以首先先要安装modescope

```shell
pip install modelscope

modelscope download --dataset gongjy/minimind_dataset pretrain_hq.jsonl --local_dir ./dataset
```

step2: 观测数据

①.在预训练之前，我们首先对原始数据有一个基本的了解，并且对其做一个基础的信息统计，可以运行tests文件夹下的dataset_test.ipynb

![image-20250520103523431](images\README\image-20250520103523431.png)

![image-20250520103559483](images\README\image-20250520103559483.png)

从数据统计中，可以看出预训练的语料只有140w行数据，并且每一行的语料平均字数只有400左右，说明语料确实经过筛选和过滤。这能够帮助小模型快速找到参数所在的空间，同时参数量较小的模型的表达能力是有限的，更适合学习数据中的简单规律，从这里分析可以看出使用小语料进行小模型的训练还是不错的。

②.在实际训练阶段，常规使用PyTorch进行训练的时候，会使用Dataset和DataLoader来读取和封装数据，喂给模型进行训练。

```python
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # 使用 tokenizer 对样本中的 'text' 字段进行编码
        # max_length: 限制最大长度为 self.max_length
        # padding='max_length': 将所有序列填充到 max_length 长度
        # truncation=True: 如果序列超过 max_length，则截断
        # return_tensors='pt': 返回 PyTorch 张量格式的结果
        encoding = self.tokenizer(
            str(sample['text']),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
         # 创建损失掩码，标记非填充位置为 True，填充位置为 False
        # 这样在计算损失时可以忽略填充位置
        input_ids = encoding.input_ids.squeeze()
        loss_mask = (input_ids != self.tokenizer.pad_token_id)
        # 输入序列 X 是 input_ids 去掉最后一个 token
        # 这是因为在自回归语言模型中，我们用前 n-1 个 token 预测后 n-1 个 token
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        # 目标序列 Y 是 input_ids 去掉第一个 token
        # 这样 X[i] 对应的预测目标就是 Y[i]，即下一个 token
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask
```

自回归语言模型原理 (加载原始文本 → 分词编码 → 截断/填充 → 创建输入-目标对 → 生成损失掩码)：

- 自回归语言模型的核心思想是使用前面的 token 序列预测下一个 token
- 通过将输入序列错位一位，创建输入-目标对： [t₁, t₂, ..., tₙ₋₁] → [t₂, t₃, ..., tₙ]
- 损失掩码用于在计算损失时忽略填充位置，只对实际内容计算损失，避免模型学习预测填充符的无意义任务

step3: 训练模型

step4: 模型测试

step5: 训练加速

- flash attention 对模型的加速
- torch.compile 对模型的加速
- 单机多卡训练
- 多机多卡训练



wandb的使用



训练扩展1: 使用更大的参数规模训练小的语料。



> 模型核心配置
>
> |      字段名       | 默认值 | 说明                                                         |
> | :---------------: | :----: | :----------------------------------------------------------- |
> |  max_seq_length   |  512   | 对模型的总参数量没有影响，但是对模型的训练，推理，内存有比较明显的影响<br />1.训练语料大于max_seq_length会被截断，小于max_seq_length时会padding<br />2.在注意力机制计算的时候，创建因果掩码矩阵的时候，会影响其大小<br />3.RoPE进行位置编码的时候，也要注意该参数的影响<br />4.如果使用KVCache也会影响推理阶段内存的占用 |
> | global_batch_size |  1024  | batch_size * K(梯度累计的步数) * GPU<br />G过大：直接导致每轮epoch的迭代次数变少，训练步数变少，在多轮epoch上不停地训练可能会过拟合该训练数据集<br />G过小：直接导致每轮epoch的迭代次数变多，训练步数变多，每次反向传播的时候噪声较多，训练不稳定，模型收敛慢。 |
> |                   |        |                                                              |
>
> 



### SFT

1.SFT阶段并不是训练的数据量越大越好，指令的多样性和复杂性更加重要。

2.在SFT阶段引入新知识容易增加模型的产生幻觉的可能。模型主要是通过预训练阶段引入新的知识，而SFT的作用是教会模型如何高效正确地利用这些数据

SFT与预训练的一个显著区别就是两者的Loss计算规则不一样





### DPO











### RAG
