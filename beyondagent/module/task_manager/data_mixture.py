from abc import ABC, abstractmethod
from typing import Sequence, List
import random
import math
from beyondagent.schema.task import Task, TaskObjective
from loguru import logger


class MixtureStrategy(ABC):
    """数据混合策略接口"""
    
    @abstractmethod
    def mix_data(self, 
                 synthetic_objectives: List[TaskObjective], 
                 original_tasks: Sequence[TaskObjective]) -> List[TaskObjective]:
        """
        混合合成数据和原始数据
        
        Args:
            synthetic_objectives: 合成的任务目标列表
            original_tasks: 原始任务列表
            
        Returns:
            混合后的任务目标列表
        """
        pass
    

class UnifiedMixtureStrategy(MixtureStrategy):
    """
    通用混合策略，可以覆盖所有常见的混合场景
    
    Examples:
        # 完全使用原始数据
        UnifiedMixtureStrategy(use_original=True, synthetic_ratio=0)
        
        # 完全使用合成数据
        UnifiedMixtureStrategy(use_original=False, synthetic_ratio=1.0)
        
        # 使用所有原始数据 + 0.5倍数量的合成数据
        UnifiedMixtureStrategy(use_original=True, synthetic_ratio=0.5)
        
        # 不使用原始数据，使用2倍原始数据数量的合成数据
        UnifiedMixtureStrategy(use_original=False, synthetic_ratio=2.0)
        
        # 使用所有原始数据 + 1.5倍数量的合成数据
        UnifiedMixtureStrategy(use_original=True, synthetic_ratio=1.5)
        
        # 使用固定种子确保可重现性
        UnifiedMixtureStrategy(use_original=True, synthetic_ratio=0.8, seed=42)
    """
    
    def __init__(self, 
                 use_original: bool = True, 
                 synthetic_ratio: float = 0.0, 
                 shuffle: bool = True,
                 seed: Optional[int] = None):
        """
        Args:
            use_original: 是否使用原始数据
                - True: 使用所有原始数据
                - False: 不使用原始数据
            synthetic_ratio: 合成数据的比例，相对于原始数据数量
                - 0: 不使用合成数据
                - 0-1: 使用 原始数据数量 * synthetic_ratio 个合成数据
                - >1: 使用 原始数据数量 * synthetic_ratio 个合成数据（多倍数据）
            shuffle: 是否在混合后打乱数据
            seed: 随机种子，用于控制抽样和shuffle的随机性
                - None: 使用当前随机状态
                - int: 使用指定的种子值
        """
        self.use_original = use_original
        self.synthetic_ratio = synthetic_ratio
        self.shuffle = shuffle
        self.seed = seed
        
        if synthetic_ratio < 0:
            raise ValueError("synthetic_ratio must be non-negative")
    
    def mix_data(self, 
                 synthetic_objectives: List[TaskObjective], 
                 original_tasks: Sequence[Task]) -> List[TaskObjective]:
        
        # 如果指定了种子，创建一个独立的随机状态
        rng = random.Random(self.seed) if self.seed is not None else random
        
        mixed_objectives = []
        
        # 处理原始数据
        if self.use_original:
            original_objectives = [
                TaskObjective(task=task, ground_truth="[env]", confidence=1.0, reward=None) 
                for task in original_tasks
            ]
            mixed_objectives.extend(original_objectives)
            logger.info(f"Added {len(original_objectives)} original tasks")
        
        # 处理合成数据
        if self.synthetic_ratio > 0:
            # 计算需要的合成数据数量
            target_synthetic_count = int(len(original_tasks) * self.synthetic_ratio)
            
            if target_synthetic_count > 0:
                if target_synthetic_count > len(synthetic_objectives):
                    # 如果需要的合成数据超过现有数量，使用所有合成数据
                    selected_synthetic = synthetic_objectives[:]
                    logger.warning(f"Not enough synthetic data: need {target_synthetic_count}, have {len(synthetic_objectives)}, using all available")
                else:
                    # 使用指定的随机状态进行采样
                    selected_synthetic = rng.sample(synthetic_objectives, target_synthetic_count)
                
                mixed_objectives.extend(selected_synthetic)
                logger.info(f"Added {len(selected_synthetic)} synthetic tasks (ratio={self.synthetic_ratio})")
        
        # 打乱数据（使用指定的随机状态）
        if self.shuffle:
            rng.shuffle(mixed_objectives)
        
        # 记录最终结果
        original_count = len(original_tasks) if self.use_original else 0
        synthetic_count = len(mixed_objectives) - original_count
        
        seed_info = f" with seed {self.seed}" if self.seed is not None else ""
        logger.info(f"Final mixture{seed_info}: {original_count} original + {synthetic_count} synthetic = {len(mixed_objectives)} total")
        
        return mixed_objectives
    
    def __repr__(self):
        return f"UnifiedMixtureStrategy(use_original={self.use_original}, synthetic_ratio={self.synthetic_ratio}, shuffle={self.shuffle}, seed={self.seed})"


class OriginalOnlyStrategy(UnifiedMixtureStrategy):
    """只使用原始数据的策略（UnifiedMixtureStrategy的别名）"""
    def __init__(self, shuffle: bool = True):
        super().__init__(use_original=True, synthetic_ratio=0.0, shuffle=shuffle)