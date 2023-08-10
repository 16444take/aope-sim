# AOPE (Automated Order Picking Environment)
Simplified simulation environment of automated order picking operation for MARL (Multi-Agent Reinforcement Learning) introduced in "[The Impact of Overall Optimization on Wareshouse Automation](https://ras.papercept.net/conferences/conferences/IROS23/program/IROS23_ContentListWeb_1.html#moat16_10)" accepted at International Conference on Intelligent Robots and Systems ([IROS2023](https://ieee-iros.org/)).

## Features
- Consists of four types of material handling equipment whose controller makes decisions for task selection (discrete actions for RL agents).
  1. FR (Flow Rack): allocates sorting boxes for collecting items. Each sorting box corresponds to a shipping destination. When all ordered items are sorted into a box, the box is transported to the next working area, and FR replaces it with a new one. The FR controller selects a shipping box to start order picking tasks for its required items.
  2. PC (Parallel Conveyor): transports items from the inventory area to allocated shipping boxes in FR. There are three conveyors in parallel, and items are loaded from the inventory area by type. Each conveyor has six loading ports. The PC controller selects an item type to be loaded on the conveyor from the inventory area.
  3. PR1 (Picking Robot 1): picks up items from the PC and places them on a carousel conveyor one by one. PR1 can also move among the conveyors to pick items from each conveyor. The PR1 controller selects a conveyor to pick items: conveyor selection occurs whenever the picking of the same item type is completed.
  4. PR2 (Picking Robot 2): picks up items from a carousel conveyor and sorts them into shipping boxes one by one. The PR2 controller selects an allocated shipping box to sort items from the carousel conveyor.

https://github.com/16444take/aope-sim/assets/140193531/597381b2-e481-40b3-8347-67da731544e7

- To prevent the simulation from being fixed to one work scenario, the item loading and replacement times of PCs and the picking time of both PR1 and PR2 are given normal distribution variations.
- Can train decentralized control policies for selecting agents' tasks with different four MARL frameworks.
  - ILLR: Indepedent Learning with Localized Reward
  - ILGR: Indepedent Learning with Globalized Reward
  - CDIC: CTDE (Centralized Training with Decentralized Execution) with Individual Critic
  - CDSC: CTDE with Shared Critic

![marl_frameworks](https://github.com/16444take/aope-sim/assets/140193531/73df3a09-6009-42fc-8a22-f692c0fb104e)

- PPO (Proximal Policy Optimization) based actor critic methods
  - [IPPO](https://arxiv.org/abs/2011.09533) (Independent PPO) for ILLR and ILGR
  - [MAPPO](https://arxiv.org/abs/2103.01955) (Multi-Agent PPO) for CDIC and CDSC
    
## Implimentation
- 100% Python code (run > 3.7.6)
- Require [Ray](https://docs.ray.io/en/latest/ray-overview/installation.html) to collect rollouts in parallel.
- Use [Pytorch](https://pytorch.org/https://pytorch.org/) for training deep neural netowrks


## Usage
### Train
```
$python train_xxxx.py 
```
- Training results can be viewed from /log_train/YYMMDD_MMHHSS/rl_metrics_XX.csv. 
