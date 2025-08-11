from beyondagent.module.advantage_assignment.advantage_analyzer import (
    load_real_training_batch,
    create_dataproto_from_batch,
    compute_three_stage_advantages,
    visualize_advantage_distributions
    )
import numpy as np
import torch
import os

def analyze_real_advantage_distribution():
    """
    分析真实数据的advantage分布
    使用你的checkpoint和训练数据
    """
    # 配置路径 - 根据你的实际路径修改
    ckpt_path = "/mnt/data/taoshuchang.tsc/beyondagent/BeyondAgent/checkpoints/w_qwen25/qwen25_14b_allstepeval_neg_stepalign/global_step_50/actor"
    config_path = "/mnt/data/taoshuchang.tsc/beyondagent/BeyondAgent/config/beyond_agent_dataflow.yaml"
    
    print("="*70)
    print("REAL DATA ADVANTAGE DISTRIBUTION ANALYSIS")
    print("="*70)
    
    try:
        # Step 1: 加载真实训练数据
        batch_dict, config, tokenizer = load_real_training_batch(ckpt_path, config_path)
        
        # Step 2: 转换为DataProto格式
        dataproto = create_dataproto_from_batch(batch_dict, config, tokenizer)
        
        # Step 3: 计算三个阶段的advantages
        original_adv, rescaled_adv, normalized_adv = compute_three_stage_advantages(dataproto, config)
        
        # Step 4: 打印详细统计
        print_detailed_statistics(original_adv, rescaled_adv, normalized_adv)
        
        # Step 5: 可视化分布
        visualize_advantage_distributions(original_adv, rescaled_adv, normalized_adv)
        
        print("\n✓ Analysis completed successfully!")
        
        return {
            'original': original_adv,
            'rescaled': rescaled_adv, 
            'normalized': normalized_adv,
            'config': config,
            'dataproto': dataproto
        }
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_detailed_statistics(original_adv: torch.Tensor, 
                            rescaled_adv: torch.Tensor, 
                            normalized_adv: torch.Tensor):
    """打印详细的统计信息"""
    
    def get_stats(tensor, name):
        values = tensor[tensor != 0].cpu().numpy()
        if len(values) == 0:
            return f"{name}: No valid values"
        
        stats = {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75)
        }
        
        return stats, values
    
    print("\n" + "="*70)
    print("DETAILED STATISTICS")
    print("="*70)
    
    orig_stats, orig_vals = get_stats(original_adv, "Original")
    resc_stats, resc_vals = get_stats(rescaled_adv, "Rescaled") 
    norm_stats, norm_vals = get_stats(normalized_adv, "Normalized")
    
    for name, stats in [("ORIGINAL GRPO", orig_stats), ("SSA RESCALED", resc_stats), ("NORMALIZED", norm_stats)]:
        if isinstance(stats, str):
            print(f"\n{name}: {stats}")
            continue
            
        print(f"\n{name}:")
        print(f"  Count:    {stats['count']}")
        print(f"  Mean:     {stats['mean']:.6f}")
        print(f"  Std:      {stats['std']:.6f}")
        print(f"  Median:   {stats['median']:.6f}")
        print(f"  Range:    [{stats['min']:.6f}, {stats['max']:.6f}]")
        print(f"  Q25-Q75:  [{stats['q25']:.6f}, {stats['q75']:.6f}]")
    
    # 计算变化
    if isinstance(orig_stats, dict) and isinstance(resc_stats, dict):
        print(f"\nSSA RESCALING EFFECTS:")
        print(f"  Mean change:   {resc_stats['mean'] - orig_stats['mean']:+.6f}")
        print(f"  Std change:    {resc_stats['std'] - orig_stats['std']:+.6f}")
        print(f"  Range change:  {(resc_stats['max'] - resc_stats['min']) - (orig_stats['max'] - orig_stats['min']):+.6f}")
    
    if isinstance(resc_stats, dict) and isinstance(norm_stats, dict):
        print(f"\nNORMALIZATION EFFECTS:")
        print(f"  Mean change:   {norm_stats['mean'] - resc_stats['mean']:+.6f}")
        print(f"  Std change:    {norm_stats['std'] - resc_stats['std']:+.6f}")
        print(f"  Range change:  {(norm_stats['max'] - norm_stats['min']) - (resc_stats['max'] - resc_stats['min']):+.6f}")
    
    if isinstance(orig_stats, dict) and isinstance(norm_stats, dict):
        print(f"\nOVERALL TRANSFORMATION:")
        print(f"  Mean change:   {norm_stats['mean'] - orig_stats['mean']:+.6f}")
        print(f"  Std change:    {norm_stats['std'] - orig_stats['std']:+.6f}")
        print(f"  Range change:  {(norm_stats['max'] - norm_stats['min']) - (orig_stats['max'] - orig_stats['min']):+.6f}")


# 简化版本：如果你已经有现成的batch
def analyze_existing_batch(batch_dict: dict, config_path: str, tokenizer_path: str):
    """
    如果你已经有现成的batch_dict，直接分析
    """
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    
    # 加载配置和tokenizer
    config = OmegaConf.load(config_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 转换和分析
    dataproto = create_dataproto_from_batch(batch_dict, config, tokenizer)
    original_adv, rescaled_adv, normalized_adv = compute_three_stage_advantages(dataproto, config)
    
    print_detailed_statistics(original_adv, rescaled_adv, normalized_adv)
    visualize_advantage_distributions(original_adv, rescaled_adv, normalized_adv)
    
    return original_adv, rescaled_adv, normalized_adv


if __name__ == "__main__":
    # 运行分析
    results = analyze_real_advantage_distribution()