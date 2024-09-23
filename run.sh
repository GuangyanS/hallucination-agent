#!/bin/bash

python run_inference.py \
    --dataset riddlesense \
    --model_path /home/gs3260/init_weights/ \
    --model Meta-Llama-3.1-8B-Instruct \
    --method zero_shot_cot \
    --sample_n 15 \
    --output_dir experiment/riddlesense_15.pkl
