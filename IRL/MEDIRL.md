# Maximum Entropy Deep Inverse Reinforcement Learning in Discovering Cyclists' Street Visual Preferences

## Brief Introduction

### Algorithms Implemented
- [Linear programming IRL. From Ng & Russell, 2000. Small state space and large state space linear programming IRL.](https://ai.stanford.edu/~ang/papers/icml00-irl.pdf)
- [Maximum entropy IRL. From Ziebart et al., 2008.](https://cdn.aaai.org/AAAI/2008/AAAI08-227.pdf)
- [Deep maximum entropy IRL. From Wulfmeier et al., 2015; original derivation.](https://arxiv.org/abs/1507.04888#)
- [Inverse-Reinforcement-Learning From Mattew Alger et al., 2017.](https://github.com/MatthewJA/Inverse-Reinforcement-Learning?tab=readme-ov-file)  
  
### My Contribution 
- **Formulate the cycling procedure as Markov Decision Process(MDP) and customize both the agent and the simulation environment for the research area**.
- **Implement MEDIRL Algorithm Using Pytorch**
  - The former implemention of MEDIRL is developed based on open source DL library **Theano**, which limits its ability to integrate advanced DL approaches such as CNNs, RNNs, and Transformers,···. 
  - **This project utilizes pytorch to implement MEDIRL, allowing for future extension**

### Reqirements
- geopandas==0.14.4
- multiset==3.1.0
- networkx==3.1
- numpy==1.26.4
- pandas==2.2.2
- Shapely==2.0.4
- Pytorch==2.4.0-CUDA11.8
  
## 1. [gridWalk.py](https://github.com/XiWen0627/MaxEnIRL_Try/blob/main/IRL/gridWalk.py)
## 2. [DPforGridBike_v1.py](https://github.com/XiWen0627/MaxEnIRL_Try/blob/main/IRL/DPforGridBike_v1.py)
## 3. [MEDIRL.ipynb](https://github.com/XiWen0627/MaxEnIRL_Try/blob/main/IRL/MEDIRL.ipynb)
