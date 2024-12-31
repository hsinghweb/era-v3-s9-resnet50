import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Tuple
import time
from tqdm import tqdm

class Trainer:
    def __init__(self, model, config, logger, checkpoint_handler):
        self.model = model
        self.config = config
        self.logger = logger
        self.checkpoint_handler = checkpoint_handler
        
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=config.base_lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
        
        self.scaler = GradScaler()
        self.best_acc = 0.0
        
    def _get_lr_scheduler(self):
        if self.config.lr_schedule_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=0
            )
        else:
            return optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.config.lr_milestones,
                gamma=self.config.lr_gamma
            )
    
    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % self.config.log_frequency == 0:
                metrics = {
                    'loss': loss.item(),
                    'accuracy': 100. * correct / total
                }
                self.logger.log_metrics(
                    metrics,
                    epoch * len(train_loader) + batch_idx,
                    prefix='train'
                )
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100. * correct / total
        }
    
    @torch.no_grad()
    def validate(self, val_loader, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(val_loader, desc='Validation'):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        metrics = {
            'loss': total_loss / len(val_loader),
            'accuracy': 100. * correct / total
        }
        
        self.logger.log_metrics(metrics, epoch, prefix='val')
        return metrics
    
    def train(self, train_loader, val_loader) -> Dict[str, float]:
        scheduler = self._get_lr_scheduler()
        
        for epoch in range(self.config.num_epochs):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader, epoch)
            
            scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % self.config.checkpoint_frequency == 0:
                state = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': self.best_acc,
                }
                
                is_best = val_metrics['accuracy'] > self.best_acc
                if is_best:
                    self.best_acc = val_metrics['accuracy']
                
                self.checkpoint_handler.save_checkpoint(
                    state,
                    epoch,
                    is_best=is_best
                )
        
        return {'best_accuracy': self.best_acc} 