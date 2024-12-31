import torch
from src.models.resnet import create_resnet50
from src.data.dataset import ImageNetDataModule
from src.trainer import Trainer
from src.utils.logger import MetricLogger
from src.utils.checkpointing import CheckpointHandler
from config import TrainingConfig
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to ImageNet dataset')
    parser.add_argument('--resume_from', type=str,
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Initialize config
    config = TrainingConfig()
    config.data_dir = args.data_dir
    
    # Set up data
    data_module = ImageNetDataModule(config)
    data_module.setup()
    train_loader, val_loader = data_module.get_dataloaders()
    
    # Create model
    model = create_resnet50(num_classes=config.num_classes)
    
    # Initialize training utilities
    logger = MetricLogger(config.log_dir)
    checkpoint_handler = CheckpointHandler(config.checkpoint_dir)
    
    # Resume from checkpoint if specified
    if args.resume_from:
        checkpoint = checkpoint_handler.load_checkpoint(args.resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize trainer
    trainer = Trainer(model, config, logger, checkpoint_handler)
    
    try:
        # Start training
        results = trainer.train(train_loader, val_loader)
        print(f"Training completed. Best validation accuracy: {results['best_accuracy']:.2f}%")
    except Exception as e:
        print(f"Training failed: {str(e)}")
    finally:
        logger.close()

if __name__ == '__main__':
    main() 