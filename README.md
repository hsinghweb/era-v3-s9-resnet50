# ResNet50 Training on ImageNet

This project implements training of ResNet50 from scratch on ImageNet-1K dataset, targeting 70% top-1 accuracy.

## Project Structure

```
project/
├── requirements.txt      # Python dependencies
└── src/
    ├── trainer.py       # Training loop implementation with mixed precision, checkpointing
    ├── models/
    │   └── resnet.py    # ResNet50 model architecture using PyTorch
    └── data/
        ├── dataset.py           # ImageNet data loading and augmentation
        └── prepare_imagenet.py  # Script to organize validation dataset
```

## File Descriptions

- `src/models/resnet.py`: Implements ResNet50 architecture using PyTorch's official implementation (without pre-trained weights)
- `src/data/dataset.py`: Handles data loading, augmentation, and preprocessing for ImageNet
- `src/data/prepare_imagenet.py`: Helper script to organize ImageNet validation set into proper directory structure
- `src/trainer.py`: Implements training loop with features like:
  - Mixed precision training (FP16)
  - Dynamic learning rate scheduling
  - Checkpointing
  - Metric logging
  - GPU optimization

## Setup and Training Instructions

### 1. EC2 Instance Setup
1. Launch an EC2 instance:
   - Recommended: p3.2xlarge (1 V100 GPU)
   - Alternative: g4dn.xlarge (cheaper but slower)
   - Use Deep Learning AMI (Ubuntu)
   - At least 200GB storage (gp3 for better performance)

2. Connect to instance:
   ```bash
   ssh -i your-key.pem ubuntu@your-instance-ip
   ```

### 2. Dataset Preparation
1. Register at [image-net.org](https://image-net.org/download-images.php)
2. Download ImageNet-1K (ILSVRC2012):
   - Training set (~138GB)
   - Validation set (~6.3GB)
   - Validation annotations

3. Transfer to EC2:
   ```bash
   scp -i your-key.pem /path/to/imagenet/* ubuntu@your-instance-ip:/path/to/data/
   ```

4. Organize validation set:
   ```bash
   python src/data/prepare_imagenet.py --val_dir /path/to/val --val_anno /path/to/validation_annotations.txt
   ```

### 3. Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd <repository-name>

# Create virtual environment
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Training
```bash
python train.py --data_dir /path/to/imagenet
```

## Important Considerations

### Performance Optimization
1. **Data Loading**:
   - Use appropriate num_workers (typically 4-8 per GPU)
   - Enable pin_memory for faster GPU transfer
   - Use appropriate batch size (256 for V100)

2. **Mixed Precision Training**:
   - Already implemented using torch.cuda.amp
   - Reduces memory usage and speeds up training

3. **Checkpointing**:
   - Save checkpoints every 5 epochs
   - Keep best model based on validation accuracy
   - Enable resume training from checkpoint

### Cost Optimization
1. **Instance Selection**:
   - Use Spot instances for ~70% cost reduction
   - Monitor spot prices across regions
   - Use g4dn.xlarge for development/testing

2. **Storage**:
   - Use gp3 volumes for better performance/cost ratio
   - Delete checkpoints except best and latest
   - Consider using S3 for long-term storage

3. **Development Process**:
   - Test pipeline on small subset first
   - Use smaller instances for code testing
   - Monitor GPU utilization (should be >90%)

### Time Optimization
1. **Data Pipeline**:
   - Pre-cache dataset on instance storage
   - Use appropriate number of workers
   - Enable prefetch factor in DataLoader

2. **Training**:
   - Use mixed precision training
   - Optimize batch size for GPU memory
   - Monitor GPU utilization and adjust parameters

3. **Monitoring**:
   - Use TensorBoard for real-time monitoring
   - Track GPU utilization with nvidia-smi
   - Monitor training/validation metrics

## Expected Resources

1. **Time**: 
   - ~10-14 days on p3.2xlarge
   - Longer on smaller instances

2. **Storage**:
   - Dataset: ~150GB
   - Checkpoints: ~20GB
   - Total recommended: 200GB

3. **Cost Estimate**:
   - On-demand p3.2xlarge: ~$1000-1400
   - Spot p3.2xlarge: ~$300-420
   - Storage: ~$20/month

## Troubleshooting

1. **Out of Memory**:
   - Reduce batch size
   - Enable gradient accumulation
   - Check for memory leaks

2. **Slow Training**:
   - Monitor GPU utilization
   - Check data loading bottlenecks
   - Verify CPU utilization

3. **Poor Convergence**:
   - Check learning rate schedule
   - Verify data augmentation
   - Monitor gradient norms

## References
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [ImageNet Dataset](https://image-net.org/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)