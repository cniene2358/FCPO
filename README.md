# Feasible Constraint Policy Optimization for Safe Reinforcement Learning

This repository contains code for the paper _"Feasible Constraint Policy Optimization for Safe Reinforcement Learning"_.




## How to run sample code
We first discuss installing the code and then discuss how to run an experiment.

### Installation

To install the experiment, please follow the Omnisafe installation guide.

Or you can just install the pip file.

```bash
pip install -r requirements.txt
```

### Run a sample code

The evaluation procedure usually takes time, in order to conduct an sample experiment you can run:

```bash
nohup python train_policy.py --algo FCPO --env-id SafetyAntVelocity-v1 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 20 > log.out &
```

here we choose to evaluate FCPO algorithm in the environment of SafetyAntVelocity using multiple threads to speed up the testing time, which empircally cost at most 5 hours.

### Code Organization
We design our algorithm and evaluate it on the Onmnisafe benchmark. The only different part is that we registry our algorithm and corresponding configuration, which you can find in the file `omnisafe/algorithms/on_policy/penanlty_admm/fcpo.py` and `omnisafe/configs/on_policy/FCPO.yaml`.
