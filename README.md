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

## Models

```
huggingface-cli login
KinGeorge
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

cd <sys_path>/hallucination-agent
mkdir init_models
huggingface-cli download --repo-type model meta-llama/Meta-Llama-3-8B-Instruct --local-dir init_models/Meta-Llama-3-8B-Instruct
```

## Quick Start

```
cd hallucination-agent
conda activate halu
sbatch run.slurm
```
