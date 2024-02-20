# Feasible Constraint Policy Optimization for Safe Reinforcement Learning

This repository contains code for the paper _"Feasible Constraint Policy Optimization for Safe Reinforcement Learning"_.


<!-- ## Main Idea -->

## How to run a sample code
We first discuss installing the code and then discuss how to run an experiment.

### Installation


To install the experiment, please follow the [Omnisafe installation guide](https://github.com/PKU-Alignment/omnisafe?tab=readme-ov-file#installation) and install it from source.

Optionally, for simplicity you can just install it with our modifired source code with pip.

```bash
pip install -r requirements.txt
```

### Q&A
During the installation, you may encounter following problem:
- ImportError: Failed to load GLFW3 shared library

    ```bash
    sudo apt-get install libglfw3
    sudo apt-get install libglfw3-dev
    ```

### Run a sample code

The evaluation procedure usually takes time, in order to conduct an sample experiment you can run:

```bash
nohup python train_policy.py --algo FCPO --env-id SafetyAntVelocity-v1 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 20 > log.out &
```

here we choose to evaluate FCPO algorithm in the environment of SafetyAntVelocity using multiple threads to speed up the testing time, which empirically cost around 5 hours.

### Code Organization
We design our algorithm and evaluate it on the Onmnisafe benchmark. The only different part is that we registry our algorithm and corresponding configuration, which you can find in the file `omnisafe/algorithms/on_policy/penanlty_admm/fcpo.py` and `omnisafe/configs/on_policy/FCPO.yaml`.


