# Zero-Order Optimization for LLM Fine-Tuning via Learnable Direction Sampling

This repository contains the code for experiments applying ZO-LDSD framework for different LLM Fine-Tuning tasks.

The code is based on the [benchmark](https://github.com/ZO-Bench)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training and Evaluation

To train and evaluate the model in the paper, run this command:

```
./run_script.sh
```

## Methods 

* `zo_rl_sgd` is ZO-SGD with LDSD-based sampling
* `zo_rl_adamm` is ZO-Adamm with LDSD-based sampling
* `zo_rl_jaguar` is Jaguar SignSGD with LDSD-based sampling
