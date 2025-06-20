### 介绍

这是一个对LLM大语言模型自己学习，训练的一个学习笔记，尽量覆盖相关的知识点，同时也会记录自己在训练mini LLM中遇到的问题。



### 模型结构

#### Attention

Attention是transformer中非常重要的计算模块，不管是Encoder模块还是Decoder模块。对如下的公式进行感性的理解还是比较重要的。

![image-20250619170230201](images/README/image-20250619170230201.png)

《BERT基础教程：Transformer大模型实战》中举了一个比较清楚的例子。“I am good”是一句话，我们可以把其转换成embedding的格式

<img src="images/README/image-20250619170820986.png" alt="image-20250619170820986" style="width:33%;" width="375"/>

通过输入矩阵，我们可以看出，矩阵的第一行表示单词I的词嵌入向量。以此类推，第二行对应单词am的词嵌入向量，第三行对应单词good的词嵌入向量。所以矩阵的维度为[句子的长度×词嵌入向量维度]。原句的长度为3，假设词嵌入向量维度为512，那么输入矩阵的维度就是[3×512]。

接下来就需要将其乘以Q,K,V三个变换矩阵。进行数据投影

<img src="images/README/image-20250619171123522.png" alt="image-20250619171123522" style="zoom:75%;" />

这边也需要一个感性的理解，可以理解同一输入映射到**不同语义空间**：

- Q 空间：聚焦“当前需要什么”（动态需求）
- K 空间：定义“我能提供什么”（静态特征）
- V 空间：保留“原始信息”（未失真内容）

接下来就是计算Q * K的转置，这边计算结果的shape就变成了[句子的长度，句子的长度]即[3,3]的矩阵，这9个数代表的含义为每一个单词与其他每一个单词的联系，可以看出每一个单词和自己的关系的数值是最高的，这也符合常理。到这一步就已经知道每一个token和其他所有token的相似度了。

<img src="images/README/image-20250619172032322.png" alt="image-20250619172032322" style="zoom:80%;" />

计算完了Q * K的转置之后，接下来就要除以K维度的平方根了，除以键值向量维度平方根的核心目的是**稳定梯度，防止 softmax 函数进入饱和区导致梯度消失**

![image-20250619174429907](images/README/image-20250619174429907.png)

<img src="images/README/image-20250619190843489.png" alt="image-20250619190843489" style="zoom:80%;" />

在没有mask机制的基础之上，最后一个行向量得到的就是加权融合后的矩阵投影。即第一行的表示token的"I"这个时候已经是融合到“am good”这2个token之后的信息，第二行就是表示“am” 融合“I”和“good”之后的token，最后一行即可以代表“good”和“I”和“am”之间的含义了。

<img src="images/README/image-20250619191026141.png" alt="image-20250619191026141" style="zoom:80%;" />

#### KVCache

LLM 生成文本时需基于历史 token 逐个预测新 token（自回归）。传统 Transformer 每次生成新 token 时需对整个输入序列（含历史 token）重新计算 **Key（K）** 和 **Value（V）** 矩阵，导致大量重复计算。这边的计算是发生在将token的embedding向量通过Q，K，V将其进行投影进行矩阵计算的时候，对结果进行缓存，这边需要特别注意的就是这个KVCache是发生在推理阶段，而不是训练阶段。推理阶段每一层的Q，K，V矩阵参数都是固定的，对于同一个token进过固定参数投影出来的结果必定是一样的，这是可以缓存的基础原理。在训练阶段，所有有梯度的参数都参与反向传播，是会变化的。所以是不能进行缓存的。

![image-20250620171119672](images/README/image-20250620171119672.png)

Q:为什么缓存的只有K,V投影矩阵的结果，而没有Q的结果

A：如上矩阵计算所示，是在忽略scale的基础上，Attention计算的逻辑。在推理阶段，因为是自回归模型，所以每次在预测next token的时候，举例来说，我们在预测第n个token的时候，我们其实是计算该矩阵第n行的计算逻辑，可以看出在计算第n行的时候，是需要依赖k1，v1, k2,v2 .....kn,vn的，所以你缓存q1,q2,q3是没有意义的，他们都是在计算自己token的时候才会需要进行投影。根本原因就是mask机制，因为mask机制，导致最后Attention最后结果的每一行的计算公式和后续token的Q投影结果没有关系。

Q:为什么只缓存K,V投影矩阵的结果，而不直接缓存Attention的前n-1已经推理完结果的矩阵结果。

A：可以缓存，但是没有必要，这样效率不高，因为你最后一行还是要计算k1，v1, k2,v2 .....kn,vn。有点重复缓存的感觉。



本质来说，KVCache是典型的空间换时间的例子，但是对于长序列来说，缓存KV也会占用大量的GPU的内存空间，接下来的GQA,MLA也是针对相关逻辑进行优化的。

#### MLA

MLA（Multi-head Latent Attention）是DeepSeek-V2/V3的核心创新，旨在**解决传统多头注意力（MHA）中KV Cache显存占用过高的问题**。主要利用如下的几个优化点

- 将KV投影到低维潜在空间存储，大幅减少Cache体积 
- 设计解耦的RoPE处理模块，解决位置编码与低秩压缩的兼容问题 
- 通过矩阵吸收技术，在推理时重组计算顺序减少显存访问。

MLA主要诞生的背景还是说解决在推理阶段KV Cache显存占用过高的问题。所以在研究MLA之前，我们也可以先

#### MoE

#### Rope

#### MTP

### Tokenizer



### 相关资源下载

|       语料描述       | 说明 | 下载地址 |
| :------------------: | :--: | :------: |
|    原始预训练语料    |      |          |
| pretrain的预训练语料 |      |          |
|                      |      |          |



### 模型预训练



1.预训练语料准备

为了得到高质量的训练语料，也是选用了匠心科技的SFT的13G左右中文问答语料，同时也融合了中文维基百科的4G左右的训练语料。因为匠心科技的SFT中有很多AI生成的范式

预训练的语料有很大的比例超过了max_seq_length



3.训练的时候遇到的实际问题

复读机问题

- 预训练的语料有很大的比例超过了max_seq_length
- 预训练的质量较低，没有进行很好的清洗，导致在很少的flops中模型学不到东西。
- 训练的flops不够，可以增加epochs训练
- model.generate中temperature的值较低。

如果我们只是训练一个小的模型，训练的语料只有几十个G左右，这种情况下，最好的解决模型推理期间出现复读机的问题，还是去增加你的清洗规则，提升你训练语料的质量，这种情况下，在训练不到1/4 epoch的时候，模型就已经初步具有回答问题的能力，不会出现大量复读的问题。





> 模型核心配置
>
> |      字段名       | 默认值 | 说明                                                         |
> | :---------------: | :----: | :----------------------------------------------------------- |
> |  max_seq_length   |  512   | 对模型的总参数量没有影响，但是对模型的训练，推理，内存有比较明显的影响<br />1.训练语料大于max_seq_length会被截断，小于max_seq_length时会padding<br />2.在注意力机制计算的时候，创建因果掩码矩阵的时候，会影响其大小<br />3.RoPE进行位置编码的时候，也要注意该参数的影响<br />4.如果使用KVCache也会影响推理阶段内存的占用 |
> | global_batch_size |  1024  | batch_size * K(梯度累计的步数) * GPU<br />G过大：直接导致每轮epoch的迭代次数变少，训练步数变少，在多轮epoch上不停地训练可能会过拟合该训练数据集<br />G过小：直接导致每轮epoch的迭代次数变多，训练步数变多，每次反向传播的时候噪声较多，训练不稳定，模型收敛慢。 |
> |                   |        |                                                              |
>



### SFT

1.SFT阶段并不是训练的数据量越大越好，指令的多样性和复杂性更加重要。

2.在SFT阶段引入新知识容易增加模型的产生幻觉的可能。模型主要是通过预训练阶段引入新的知识，而SFT的作用是教会模型如何高效正确地利用这些数据

SFT与预训练的一个显著区别就是两者的Loss计算规则不一样。

#### SFT数据集





### DPO(Direct Preference Optimization 直接偏好对齐)

#### 基本原理

大模型（如 ChatGPT）训练完后，可能输出 **不符合人类偏好** 的内容，比如：

- 答案冗长啰嗦 😤
- 包含错误或偏见 ❌
- 忽略关键问题 🙅

传统方法 **RLHF**（基于人类反馈的强化学习）能对齐偏好，但流程复杂：

1. 需额外训练一个 **奖励模型**（Reward Model）打分；
2. 再用强化学习（如 PPO）微调模型，过程不稳定且计算成本高。

👉 **DPO 的目标**：**跳过奖励模型**，直接用偏好数据微调模型，**更加简单高效地进行训练**！



### RL（强化学习）

策略迭代算法 VS 价值迭代算法

#### **一、策略迭代算法（Policy Iteration）**

**核心思想**：通过交替进行 **策略评估（Policy Evaluation）** 和 **策略改进（Policy Improvement）**，逐步优化策略直至收敛到最优策略

**适用场景**：状态空间较小的问题（如棋盘游戏、简单机器人控制）。

**具体步骤**：

1. **策略评估（Policy Evaluation）**

   - 固定当前策略 πk，通过贝尔曼方程迭代计算其状态价值函数 Vπk(s)：

     ![image-20250607110044712](images/README/image-20250607110044712.png)

   - 重复计算直至 V(s) 收敛（误差小于阈值）。

2. **策略改进（Policy Improvement）**

   - 基于当前 V(s)，对每个状态选择最大化动作价值 Q(s,a) 的动作：

     ![image-20250607110147283](images/README/image-20250607110147283.png)

   - 新策略 πk+1 是**贪婪策略**（只选当前最优动作）。

3. **循环迭代**：重复评估与改进，直到策略不再变化（πk+1=πk）。

**特点**：

- ✅ **优点**：策略序列单调改进，收敛稳定；
- ❌ **缺点**：策略评估需完全收敛，计算成本高（尤其大规模状态空间）。

#### **二、价值迭代算法（Value Iteration）**

**核心思想**：跳过显式策略，**直接优化状态价值函数 V(s)**，通过贝尔曼最优方程一步合并策略评估与改进

**适用场景**：大规模状态空间问题（如复杂路径规划）。

具体步骤：

1. **价值更新（Value Update）**

   - 直接迭代更新状态价值至最优值 V∗(s)：

     ![image-20250607110512862](images/README/image-20250607110512862.png)

   - 无需等待策略评估收敛，每次更新即隐含策略改进。

2. **策略提取（Policy Extraction）**

   - 收敛后，通过最优价值函数导出最优策略：

     ![image-20250607110454969](images/README/image-20250607110454969.png)

**特点**：

- ✅ **优点**：计算效率高（省去策略评估的多次迭代）；
- ❌ **缺点**：中间过程无显式策略，收敛前策略不可用。

#### **三、核心区别与联系**

| **维度**     | **策略迭代**                                  | **价值迭代**                   |
| :----------- | :-------------------------------------------- | :----------------------------- |
| **原理**     | 显式优化策略 π                                | 隐式优化值函数 V(s) → 导出策略 |
| **流程**     | 评估+改进交替                                 | 直接更新 V(s)（合并两步）      |
| **计算成本** | 高（需策略评估收敛）                          | 低（单次更新包含策略改进）     |
| **收敛速度** | 策略收敛快（少轮次）                          | 值函数收敛慢（多轮次）         |
| **适用性**   | 中小状态空间                                  | 大规模状态空间                 |
| **本质关联** | 价值迭代是策略迭代的极限简化（策略评估仅1次） |                                |

### RAG



#### Embedding



#### Reranker



#### Faiss
