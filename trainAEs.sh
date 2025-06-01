#!/bin/bash

CONFIGS=(
    "exp/TMNIST_AE/config.json" \
    "exp/TMNIST_CACTUS/config.json" \
    "exp/GIVECREDIT_AE/config.json" \
    "exp/GIVECREDIT_CACTUS/config.json" 
)


echo "Starting training"

for CONFIG in "${CONFIGS[@]}"; do
    echo "training $CONFIG"
    python train.py --config $CONFIG
done



