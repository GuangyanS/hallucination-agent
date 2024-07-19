# hallucination agent

## Requirements

```
conda create -n halu python=3.10 -y
conda activate halu

git clone https://github.com/GuangyanS/hallucination-agent
cd hallucination-agent
pip install --upgrade pip 
pip install -r requirements.txt
```

## Datasets

Download the datasets from the following:

```
https://github.com/kojima-takeshi188/zero_shot_cot/tree/main/dataset
```

download this under hallucination-agent so we have `hallucination-agent/dataset/xxx`

## Models

```
huggingface-cli download --repo-type model meta-llama/Meta-Llama-3-8B-Instruct --local-dir <sys_path>/hallucination-agent/init_weights/Meta-Llama-3-8B-Instruct
```

## Quick Start

```
cd hallucination-agent
conda activate halu
sbatch run.slurm
```
