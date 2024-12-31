from dataclasses import dataclass
from typing import Tuple

@dataclass
class TrainingConfig:
    # Dataset parameters
    data_dir: str = '/path/to/imagenet'
    num_classes: int = 1000
    input_size: Tuple[int, int] = (224, 224)
    
    # Training parameters
    batch_size: int = 256
    num_epochs: int = 100
    base_lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    
    # Learning rate schedule
    lr_schedule_type: str = 'cosine'  # ['step', 'cosine']
    lr_milestones: list = None
    lr_gamma: float = 0.1
    warmup_epochs: int = 5
    
    # Hardware
    num_workers: int = 8
    device: str = 'cuda'
    
    # Checkpointing
    checkpoint_dir: str = './checkpoints'
    checkpoint_frequency: int = 5
    
    # Logging
    log_dir: str = './logs'
    log_frequency: int = 100 