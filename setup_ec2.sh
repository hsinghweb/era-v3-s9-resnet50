#!/bin/bash

# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y \
    build-essential \
    git \
    python3-pip \
    python3-dev \
    nvidia-driver-525 \
    nvidia-cuda-toolkit

# Create a virtual environment
python3 -m venv env
source env/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p checkpoints logs data

echo "Setup complete! Make sure to:"
echo "1. Download ImageNet dataset to ./data"
echo "2. Activate the virtual environment: source env/bin/activate"
echo "3. Run training: python train.py --data_dir ./data" 