import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import math
import warnings
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import PretrainDataset
import inspect
from typing import Tuple

warnings.filterwarnings('ignore')


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def get_lr_with_warmup(it, args):
    """
    根据迭代次数返回学习率, it为总迭代次数
    """
    max_lr = args.max_lr  # 最大学习率
    min_lr = args.min_lr  # 最小学习率
    warmup_iters = args.warmup_iters  # 预热迭代次数
    lr_decay_iters = args.lr_decay_iters  # 衰减迭代次数

    # 1. warmup 阶段
    if it < warmup_iters:
        return max_lr * it / warmup_iters  # 线性增加到最大学习率
    # 2. 衰减结束，使用最小学习率
    if it > lr_decay_iters:
        return min_lr  # 衰减结束，使用最小学习率
    # 3. 余弦衰减阶段
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)  # 衰减阶段中，当前迭代相对于剩余迭代的比例
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff 是一个从 0 到 1 之间变化的系数，控制学习率的衰减
    return min_lr + coeff * (max_lr - min_lr)

def train_epoch(epoch, wandb):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()

    # 新增：初始化token计数器和处理开始时间
    total_tokens = 0
    tokens_start_time = time.time()

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        #lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        it = epoch * iter_per_epoch + step + 1  # 当前全局迭代次数
        lr = get_lr_with_warmup(it, args)  # 调用新的学习率函数
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        # 新增：计算当前批次的token数量
        batch_tokens = loss_mask.sum().item()

        # 新增：在分布式环境下汇总所有进程的token数
        if ddp:
            batch_tokens_tensor = torch.tensor(batch_tokens, dtype=torch.float, device=args.device)
            dist.all_reduce(batch_tokens_tensor, op=dist.ReduceOp.SUM)
            batch_tokens = batch_tokens_tensor.item()

        # 新增：累加token计数
        total_tokens += batch_tokens

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time

            # 新增：计算每秒处理的token数
            elapsed_time = time.time() - tokens_start_time
            tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0

            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min tokens/sec:{:.1f}'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                    tokens_per_second))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                           "tokens_per_sec": tokens_per_second})
            tokens_start_time = time.time()
            total_tokens = 0

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{args.num_hidden_layers}_{lm_config.hidden_size}{moe_path}_{epoch}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            model.train()

    # 新增：打印整个epoch的平均token处理速度
    if not ddp or dist.get_rank() == 0:
        total_elapsed_time = time.time() - tokens_start_time
        avg_tokens_per_second = total_tokens / total_elapsed_time if total_elapsed_time > 0 else 0
        print(f"Epoch {epoch + 1} average tokens/sec: {avg_tokens_per_second:.1f}")


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    model = MiniMindForCausalLM(lm_config).to(args.device)
    Logger(f'LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    # 只对内部的MiniMindModel进行编译，而不是整个MiniMindForCausalLM
    if args.use_torch_compile and hasattr(torch, 'compile') and torch.__version__ >= '2.0.0':
        Logger("仅对MiniMindModel进行torch.compile加速")
        model.model = torch.compile(model.model)
    return model, tokenizer


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


def configure_optimizer(model, weight_decay: float, learning_rate: float, betas: Tuple[float, float], device_type: str='cuda'):
    """
    配置 AdamW 优化器, 并对参数分组, 以应用不同的优化策略, 通常权重矩阵(2D及以上)应用权重衰减, 而偏置(bias)和层归一化(LayerNorm)的参数不应用权重衰减

    Args:
        weight_decay (float): 权重衰减系数
        learning_rate (float): 学习率
        betas (Tuple[float, float]): AdamW 优化器的 beta1 和 beta2 参数
        device_type (str): 设备类型, 用于指定优化器的设备

    Returns:
        torch.optim.AdamW: 优化器
    """
    # 获取模型参数并过滤不需要梯度的参数
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    # 维度大于等于 2 的参数（如权重矩阵、嵌入层参数），这些参数会应用权重衰减
    # 这些参数通常是模型的主要可学习参数，直接影响模型的表达能力
    # 维度小于 2 的参数（如偏置、LayerNorm 参数），这些参数不会应用权重衰减
    # 这些参数通常用于调整模型的输出分布，而不是直接参与特征变换
    decay_params = []
    no_decay_params = []

    for name, param in param_dict.items():
        if param.dim() < 2 or "bias" in name or isinstance(param, torch.nn.LayerNorm):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    # 创建优化器参数组
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

    # 检查是否支持融合 AdamW
    # 融合 AdamW（Fused AdamW） 是 PyTorch 提供的一种优化 AdamW 实现的高性能版本，通过将多个操作融合为一个内核（kernel）来加速计算
    # 它特别适用于 GPU 上的大规模深度学习训练任务
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

    return optimizer

# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="../out")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--use_torch_compile", default=False, type=bool)
    parser.add_argument("--use_flash_attn", default=False, type=bool)

    parser.add_argument("--weight_decay", type=float, default=1e-1)
    parser.add_argument("--max_lr", type=float, default=3e-4, help="最大学习率")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="最小学习率")
    parser.add_argument("--warmup_iters", type=int, default=None, help="预热迭代次数")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="预热迭代比例")
    parser.add_argument("--lr_decay_iters", type=int, default=None, help="学习率衰减迭代次数")
    parser.add_argument("--lr_decay_ratio", type=float, default=0.98, help="学习率衰减迭代比例")

    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl")
    args = parser.parse_args()

    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               use_moe=args.use_moe, flash_attn=args.use_flash_attn)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast('cuda')

    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"

    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        # 同时设置 CUDA 的随机种子
        torch.cuda.manual_seed(base_seed + rank)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    model, tokenizer = init_model(lm_config)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    #使用bfloat16混合精度进行训练
    scaler = torch.amp.GradScaler('cuda',enabled=(args.dtype in ['float16', 'bfloat16']))

    betas: Tuple[float, float] = (0.9, 0.95)
    # optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    optimizer = configure_optimizer(
        model,
        weight_decay=args.weight_decay,
        learning_rate=args.max_lr,
        betas=betas,
        device_type='cuda'
    )
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    total_iters = args.epochs * iter_per_epoch

    if args.warmup_iters is None:
        args.warmup_iters = int(total_iters * args.warmup_ratio)  # 预热迭代次数
    if args.lr_decay_iters is None:
        lr_decay_iters = int(total_iters * args.lr_decay_ratio)  # 衰减迭代次数
    else:
        lr_decay_iters = args.lr_decay_iters
        assert lr_decay_iters > args.warmup_iters, "衰减迭代次数必须大于预热迭代次数"
        assert lr_decay_iters <= total_iters, "衰减迭代次数必须小于总迭代次数"
        

    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
