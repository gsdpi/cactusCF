#!/bin/bash

CONFIG="exp/CF_EVALUATION/config.json"

echo "Starting CF generation"
python CFResults.py --config $CONFIG --N 100 --N_BOOTSTRAP 5 --store




