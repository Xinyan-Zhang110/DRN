# Optimal Status Update for Caching Enabled IoT Networks: A Dueling Deep R-Network Approach

## Note
File structure: 
1. `AoI_Energy.py` \
   code for environment, which simulates the status update procedure of the concerned IoT network.
2. `dqn.py` \
   code for DQN-based algorithms, i.e., DQ-DSU and DDQ-DSU.
3. `drn.py` \
   code for DRN-based algorithms, i.e., DR-DSU and DDR-DSU.

## Run an experiment
Run **DQ-DSU** in environment with $\gamma=0.95$, $N=24$, $\beta_2=1$ and $P_n=0.6$
``` bash
python dqn.py --alg dqn --gamma 0.95 --User 24 --Beta2 1 --Request_P 0.6
```

Run **DDQ-DSU** in environment with $\gamma=0.95$, $N=24$, $\beta_2=1$ and $P_n=0.6$
``` bash
python dqn.py --alg due --gamma 0.95 --User 24 --Beta2 1 --Request_P 0.6
```

Run **DR-DSU** in environment with $N=24$, $\beta_2=1$ and $P_n=0.6$
``` bash
python drn.py --alg dqn --User 24 --Beta2 1 --Request_P 0.6
```

Run **DDR-DSU** in environment with $N=24$, $\beta_2=1$ and $P_n=0.6$
``` bash
python drn.py --alg due --User 24 --Beta2 1 --Request_P 0.6
```

Evaluation results during training  will be recorded in a tensorboard log file in the current directory.

## Citing 
*C. Xu, Y. Xie, X. Wang, H. H. Yang, D. Niyato, and T. Q. S. Quek, “Optimal Status Update for Caching Enabled IoT Networks: A Dueling Deep R-Network Approach,” Jun. 2021, arXiv:2106.06945.*