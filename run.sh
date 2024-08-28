#!/bin/bash

python /home/tw9146/gysun/hallucination-agent/run_inference.py \
    --dataset strategyqa \
    --model Meta-Llama-3.1-8B-Instruct \
    --method zero_shot \
    --output_dir /home/tw9146/gysun/hallucination-agent/experiment/strategyqa.pkl
