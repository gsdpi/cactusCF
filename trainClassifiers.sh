#!/bin/bash

CONFIGS=("exp/TMNIST_class/config.json" "exp/GIVECREDIT_class/config.json")

echo "Starting training"

for CONFIG in "${CONFIGS[@]}"; do
    echo "training $CONFIG"
    python train.py --config $CONFIG
done



