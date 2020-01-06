# Neural Logic Reinforcement Learing
Implementaion of [Neural Logic Reinforcement learning](https://arxiv.org/abs/1904.10729) and several benchmarks.
Neural Logic Reinforcement Learning uses deep reinforcement leanring methods to train a differential indutive logic progamming architecture, obtaining explainable and generalizable policies. Paper accepted by ICML2019.


## Enviornments
Developed in python2.7, Linux enviornment.

### Dependencies
* numpy
* tensorflow (1.11)

## User Guide
* use main.py to run the experiments
* `--mode=` to specify the running mode, can be "train" or "generalize", where generalize means to run a generalization test.
* `--task=` to specify the task, can be  "stack", "unstack", "on" or "cliffwalking".
* `--algo` to specify agent type, can be "DILP", "NN" or "Random"
* `--name` to specify the id of this run.
* for example: `python main.py --mode=train --algo=DILP --task=unstack --name=ICMLtest`
