import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from transformers import AutoTokenizer
from omegaconf import OmegaConf
from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_advantage

from verl.utils.dataset.rl_dataset import collate_fn
from torch.utils.data import DataLoader

def load_real_training_batch(ckpt_path: str, config_path: str) -> tuple:
    """
    加载真实的训练batch
    
    Args:
        ckpt_path: checkpoint路径
        config_path: 配置文件路径 (beyond_agent_dataflow.yaml)
    
    Returns:
        (batch_dict, config, tokenizer): 真实训练batch，配置，tokenizer
    """
    print("Loading real training configuration and data...")
    
    # 1. 加载配置
    config = OmegaConf.load(config_path)
    
    # 2. 从你的脚本参数覆盖配置 (模拟你的运行脚本)
    config.data.train_files = "/mnt/data/yunpeng.zyp/data/appworld_verl/train.parquet"
    config.data.val_files = "/mnt/data/yunpeng.zyp/data/appworld_verl/dev.parquet"
    config.data.train_batch_size = 32
    config.data.max_prompt_length = 4096
    config.data.max_response_length = 20480
    config.actor_rollout_ref.model.path = "/mnt/data/zouanni.zan/models/Qwen2.5-14B-Instruct"
    
    # 3. 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 4. 创建数据集 (使用你的函数)
    from beyondagent.main_ppo import create_rl_dataset, create_rl_sampler
    
    train_dataset = create_rl_dataset(
        data_paths=config.data.train_files, 
        data_config=config.data, 
        tokenizer=tokenizer, 
        processor=None
    )
    
    train_sampler = create_rl_sampler(config.data, train_dataset)
    
    # 5. 创建dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.data.train_batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=0,  # 避免多进程问题
        pin_memory=False
    )
    
    # 6. 获取一个batch
    print("Loading first training batch...")
    batch_dict = next(iter(train_dataloader))
    
    print(f"✓ Loaded batch with {len(batch_dict)} samples")
    print(f"✓ Batch keys: {list(batch_dict.keys())}")
    
    return batch_dict, config, tokenizer


def create_dataproto_from_batch(batch_dict: dict, config, tokenizer) -> DataProto:
    """
    将batch_dict转换为DataProto，模拟env_manager.rollout()的输出
    """
    print("Converting batch_dict to DataProto format...")
    
    # 这里需要模拟你的env_manager.rollout()过程
    # 实际上应该是：batch -> tasks -> env.rollout() -> trajectories -> dataproto
    
    # 从batch_dict中提取基本信息
    batch_size = len(batch_dict)
    
    # 模拟environment rollout结果 (这部分需要你的环境交互)
    # 由于我们现在只是分析advantage分布，可以使用mock的reward
    mock_rewards = torch.rand(batch_size) * 0.8 + 0.1  # [0.1, 0.9]
    
    # 构造基本的DataProto结构
    # 这里需要根据你的实际数据格式调整
    prompts = []
    responses = []
    attention_masks = []
    step_ids = []
    steps_data = []
    reward_scores = []
    
    for i in range(batch_size):
        # 从batch_dict提取prompt
        if 'input_ids' in batch_dict[i]:
            full_ids = batch_dict[i]['input_ids']
        else:
            # 需要从raw_prompt构建
            messages = batch_dict[i].get('messages', [])
            if messages:
                prompt_text = tokenizer.apply_chat_template(
                    messages[:1], tokenize=False, add_generation_prompt=True
                )
                prompt_ids = tokenizer.encode(prompt_text)
                
                # 模拟response (这里应该是environment交互的结果)
                # 暂时使用一个简单的mock response
                mock_response = "I'll help you with this task. Let me search for the information."
                response_ids = tokenizer.encode(mock_response)
                
                full_ids = prompt_ids + response_ids
            else:
                continue
        
        # 分割prompt和response
        # 这里需要根据你的实际格式调整
        prompt_len = min(len(full_ids) // 2, config.data.max_prompt_length)
        prompt_ids = full_ids[:prompt_len]
        response_ids = full_ids[prompt_len:prompt_len + config.data.max_response_length]
        
        # 解析steps
        from beyondagent.utils.step_parser import parse_response_ids_to_steps
        try:
            parse_result = parse_response_ids_to_steps(response_ids, tokenizer)
            step_ids_list = parse_result.step_ids
            steps_list = [
                {"action": s["action_text"], "observation": s["observation_text"]} 
                for s in parse_result.steps
            ]
        except:
            # 如果解析失败，使用简单的step结构
            step_ids_list = [0] * len(response_ids)
            steps_list = [{"action": "mock action", "observation": "mock observation"}]
        
        prompts.append(prompt_ids)
        responses.append(response_ids)
        step_ids.append(step_ids_list)
        steps_data.append(steps_list)
        reward_scores.append({"outcome": mock_rewards[i].item()})
    
    # 转换为tensor并padding
    from torch.nn.utils.rnn import pad_sequence
    from verl.utils.torch_functional import pad_sequence_to_length
    
    prompts_tensor = pad_sequence(
        [torch.tensor(p) for p in prompts], 
        batch_first=True, 
        padding_value=tokenizer.pad_token_id
    )
    prompts_tensor = pad_sequence_to_length(
        prompts_tensor, 
        config.data.max_prompt_length, 
        tokenizer.pad_token_id,
        left_pad=True
    )
    
    responses_tensor = pad_sequence(
        [torch.tensor(r) for r in responses], 
        batch_first=True, 
        padding_value=tokenizer.pad_token_id
    )
    responses_tensor = pad_sequence_to_length(
        responses_tensor, 
        config.data.max_response_length, 
        tokenizer.pad_token_id
    )
    
    step_ids_tensor = pad_sequence(
        [torch.tensor(s) for s in step_ids], 
        batch_first=True, 
        padding_value=-1
    )
    step_ids_tensor = pad_sequence_to_length(
        step_ids_tensor, 
        config.data.max_response_length, 
        -1
    )
    
    # 创建attention mask和loss mask
    full_input_ids = torch.cat([prompts_tensor, responses_tensor], dim=1)
    attention_mask = (full_input_ids != tokenizer.pad_token_id).long()
    
    prompt_loss_mask = torch.zeros_like(prompts_tensor)
    response_loss_mask = (responses_tensor != tokenizer.pad_token_id).long()
    loss_mask = torch.cat([prompt_loss_mask, response_loss_mask], dim=1)
    
    # 构建batch字典
    batch = {
        "prompts": prompts_tensor,
        "responses": responses_tensor,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "step_ids": step_ids_tensor,
    }
    
    non_tensor_batch = {
        "steps": np.array(steps_data, dtype=object),
        "reward_scores": reward_scores
    }
    
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


def compute_three_stage_advantages(dataproto: DataProto, config) -> tuple:
    """
    计算三个阶段的advantages:
    1. 原始GRPO advantages
    2. SSA rescaling后的advantages  
    3. advantage normalization后的advantages
    """
    print("Computing advantages at three stages...")
    
    # Step 1: 计算token-level rewards
    reward_scores = [item["outcome"] for item in dataproto.non_tensor_batch["reward_scores"]]
    bs, response_len = dataproto.batch["responses"].shape
    
    reward_tensor = torch.zeros((bs, response_len), dtype=torch.float32)
    
    # 将reward放在sequence末尾
    prompt_len = dataproto.batch["prompts"].shape[1]
    response_mask = dataproto.batch["attention_mask"][:, prompt_len:]
    
    for i, final_reward in enumerate(reward_scores):
        valid_positions = (response_mask[i] == 1).nonzero(as_tuple=True)[0]
        if len(valid_positions) > 0:
            last_pos = valid_positions[-1].item()
            reward_tensor[i, last_pos] = final_reward
    
    dataproto.batch["token_level_rewards"] = reward_tensor
    dataproto.batch["response_mask"] = response_mask
    
    # Mock values (normally from critic)
    values = torch.zeros((bs, response_len + 1), dtype=torch.float32)
    dataproto.batch["values"] = values
    
    # Step 2: 计算原始advantages
    original_batch = DataProto(
        batch={k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in dataproto.batch.items()},
        non_tensor_batch=dataproto.non_tensor_batch.copy()
    )
    
    compute_advantage(
        original_batch,
        adv_estimator="GRPO",
        gamma=0.99,
        lam=0.95,
        num_repeat=1,
        norm_adv_by_std_in_grpo=True,
        multi_turn=False,
        config=getattr(config, 'algorithm', {})
    )
    
    original_advantages = original_batch.batch["advantages"].clone()
    
    # Step 3: 应用SSA rescaling (模拟你的语义评估逻辑)
    rescaled_advantages = apply_ssa_rescaling(
        original_advantages.clone(), 
        dataproto, 
        config.semantic_advantage
    )
    
    # Step 4: 应用advantage normalization
    normalized_advantages = apply_advantage_normalization(
        rescaled_advantages.clone(),
        dataproto,
        config.semantic_advantage.adv_norm
    )
    
    return original_advantages, rescaled_advantages, normalized_advantages


def apply_ssa_rescaling(advantages: torch.Tensor, dataproto: DataProto, semantic_config) -> torch.Tensor:
    """应用SSA rescaling"""
    print("Applying SSA rescaling...")
    
    step_ids = dataproto.batch["step_ids"]
    response_mask = dataproto.batch["response_mask"]
    
    # 模拟语义评估结果
    np.random.seed(42)
    step_flags = []
    for sample_steps in dataproto.non_tensor_batch["steps"]:
        sample_flags = []
        for step in sample_steps:
            # 简单启发式：较长的action更可能是good
            action_length = len(step.get("action", ""))
            is_good = action_length > 10 or np.random.random() > 0.4
            sample_flags.append(is_good)
        step_flags.append(sample_flags)
    
    # 应用rescaling
    scale = torch.ones_like(advantages)
    
    for b in range(len(advantages)):
        if not step_flags[b]:
            continue
            
        # 计算overall advantage
        sample_mask = response_mask[b]
        if sample_mask.sum() > 0:
            non_zero_mask = (advantages[b] != 0) & sample_mask.bool()
            if non_zero_mask.any():
                overall_adv = advantages[b][non_zero_mask].mean().item()
            else:
                overall_adv = 0.0
        else:
            overall_adv = 0.0
        
        current_step_flags = step_flags[b]
        sample_step_ids = step_ids[b]
        
        for step_id, is_good in enumerate(current_step_flags):
            step_mask = (sample_step_ids == step_id)
            if not step_mask.any():
                continue
            
            if overall_adv > 0:
                factor = semantic_config.consistent_scale if is_good else semantic_config.pos_unconsistent_scale
            else:
                factor = semantic_config.neg_unconsistent_scale if is_good else semantic_config.consistent_scale
            
            scale[b].masked_fill_(step_mask, factor)
    
    # 处理padding
    padding_mask = (step_ids == -1)
    scale.masked_fill_(padding_mask, 1.0)
    
    return advantages * scale


def apply_advantage_normalization(advantages: torch.Tensor, dataproto: DataProto, adv_norm_config) -> torch.Tensor:
    """应用advantage normalization"""
    if not adv_norm_config.enable:
        return advantages
    
    print("Applying advantage normalization...")
    
    response_mask = dataproto.batch["response_mask"]
    nonzero_mask = response_mask & (advantages != 0)
    
    if nonzero_mask.any():
        nonzero_advs = advantages[nonzero_mask]
        median = torch.median(nonzero_advs)
        std = nonzero_advs.std(unbiased=False).clamp_min(1e-8)
        
        normalized_advantages = advantages.clone()
        normalized_advantages[nonzero_mask] = (advantages[nonzero_mask] - median) / std
    else:
        normalized_advantages = advantages.clone()
    
    return normalized_advantages


def visualize_advantage_distributions(original_adv: torch.Tensor, 
                                    rescaled_adv: torch.Tensor,
                                    normalized_adv: torch.Tensor,
                                    save_path: str = "advantage_analysis_real.png"):
    """可视化三个阶段的advantage分布"""
    print("Creating visualization...")
    
    def extract_values(tensor):
        return tensor[tensor != 0].cpu().numpy()
    
    original_vals = extract_values(original_adv)
    rescaled_vals = extract_values(rescaled_adv)
    normalized_vals = extract_values(normalized_adv)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Real Data: SSA Advantage Distribution Analysis', fontsize=16, fontweight='bold')
    
    datasets = [
        ("Original GRPO", original_vals, "skyblue"),
        ("SSA Rescaled", rescaled_vals, "lightcoral"),
        ("Normalized", normalized_vals, "lightgreen")
    ]
    
    # 上排：直方图
    for i, (title, values, color) in enumerate(datasets):
        ax = axes[0, i]
        if len(values) > 0:
            ax.hist(values, bins=30, alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
            ax.set_title(f'{title}\n(n={len(values)})', fontweight='bold')
            ax.set_xlabel('Advantage Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # 统计信息
            mean_val = np.mean(values)
            std_val = np.std(values)
            median_val = np.median(values)
            
            stats_text = f'μ={mean_val:.3f}\nσ={std_val:.3f}\nmed={median_val:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 下排：对比图
    # 箱线图
    ax_box = axes[1, 0]
    valid_datasets = [(title, values) for title, values, _ in datasets if len(values) > 0]
    if valid_datasets:
        box_data = [values for _, values in valid_datasets]
        box_labels = [title for title, _ in valid_datasets]
        
        bp = ax_box.boxplot(box_data, labels=box_labels, patch_artist=True)
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax_box.set_title('Box Plot Comparison', fontweight='bold')
        ax_box.set_ylabel('Advantage Value')
        ax_box.grid(True, alpha=0.3)
        ax_box.tick_params(axis='x', rotation=45)
    
    # CDF图
    ax_cdf = axes[1, 1]
    for title, values, color in datasets:
        if len(values) > 0:
            sorted_values = np.sort(values)
            y = np.arange(1, len(values) + 1) / len(values)
            ax_cdf.plot(sorted_values, y, label=title, color=color, linewidth=2)
    
    ax_cdf.set_title('Cumulative Distribution', fontweight='bold')
    ax_cdf.set_xlabel('Advantage Value')
    ax_cdf.set_ylabel('Cumulative Probability')
    ax_cdf.legend()
    ax_cdf.grid(True, alpha=0.3)
    
    # 统计总结
    ax_stats = axes[1, 2]
    ax_stats.axis('off')
    
    stats_text = "Statistical Summary:\n\n"
    for title, values, _ in datasets:
        if len(values) > 0:
            stats_text += f"{title}:\n"
            stats_text += f"  Mean: {np.mean(values):.4f}\n"
            stats_text += f"  Std:  {np.std(values):.4f}\n"
            stats_text += f"  Min:  {np.min(values):.4f}\n"
            stats_text += f"  Max:  {np.max(values):.4f}\n\n"
    
    ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.show()
    
    return fig