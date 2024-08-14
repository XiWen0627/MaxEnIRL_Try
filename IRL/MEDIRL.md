# Maximum Entropy Deep Inverse Reinforcement Learning in Discovering Cyclists' Street Visual Preferences

## Algorithms Implemented
- [Linear programming IRL. From Ng & Russell, 2000. Small state space and large state space linear programming IRL.](https://ai.stanford.edu/~ang/papers/icml00-irl.pdf)
- [Maximum entropy IRL. From Ziebart et al., 2008.](https://cdn.aaai.org/AAAI/2008/AAAI08-227.pdf)
- [Deep maximum entropy IRL. From Wulfmeier et al., 2015; original derivation.](https://arxiv.org/abs/1507.04888#)
- [Inverse-Reinforcement-Learning From Mattew Alger et al., 2017.](https://github.com/MatthewJA/Inverse-Reinforcement-Learning?tab=readme-ov-file)  
  
## My Contribution 
- **Formulate the cycling procedure as Markov Decision Process(MDP) and customize both the agent and the simulation environment for the research area**.
- **Implement MEDIRL Algorithm Using Pytorch**
  - The former implemention of MEDIRL is developed based on open source DL library **Theano**, which limits its ability to integrate advanced DL approaches such as CNNs, RNNs, and Transformers,···. 
  - **This project utilizes pytorch to implement MEDIRL, allowing for future extension**

## Reqirements
- geopandas==0.14.4
- multiset==3.1.0
- networkx==3.1
- numpy==1.26.4
- pandas==2.2.2
- Shapely==2.0.4
- Pytorch==2.4.0-CUDA11.8

## Project Documentation
## 1. [gridWalk.py](https://github.com/XiWen0627/MaxEnIRL_Try/blob/main/IRL/gridWalk.py)
### Intuition
Cycling route decision process can be mathematically modeled as an MDP, which forms the foundation for applying IRL. In our study, **cyclists are treated as agents in an MDP**. Solving this MDP model will give us the optimal decision strategy for each different location. Specifically, we can define the elements of the MDP as follows.  
- **State**: Each state ***s∈S*** is a vector used to describe the basis of a cyclist’s decision-making, denoted as ***s=\{Dest,Pos,SE\}***. The vector Dest is the target location of current cyclist. The vector ***Pos*** is utilized to represented the road segments and the 100m grid units where cyclist is located. In other words, the location of cyclists is determined by both road segment and their corresponding grid. The vector ***SE*** comprises the semantic elements of SVIs, representing cyclists’ street visual environments at their locations.  
- **Action**: An action a∈A indicates the grid-to-grid movement choice under the restriction of road networks. Inspired by existing studies, we define a global action space ***A*** consisting of 9 movement directions — forward (F), forward left (FL), left (L), backward left (BL), backward (B), backward right (BR), right (R), forward right (FR), and stay(ST). Note that, although these 9 directions represent a comprehensive set of all potential actions to be taken anywhere, only a subset of them are applicable for most states. In order to account for the specific layout of the local network, we have also defined a local action space ***A_s∈A*** to capture all valid actions at each state ***s***.   
- **Policy**: The policy ***π(a|s)*** describes how cyclists make route decisions under the influence of the street visual environment. The optimal policy, denoted as ***π^***, represents the most representative route decision pattern for cyclists.  
- **Reward function**: The reward function ***R(s,a)*** characterizes cyclists' preferences for the street visual environment in their cycling procedures. We use a set of parameters ***θ*** to approximate the feedback for specific actions at each state, thereby replacing the location-based reward representation with a mapping between states, actions, and their associated rewards.

<table align="center" width="100%">
  <tr>
    <td><img src="https://github.com/user-attachments/assets/34906158-365d-45dd-ab0a-f874bece50f0" width="1000" alt="Figure 1" /></td>
  </tr>
  <tr align="center">
    <td><em>Process of MDP Formulation</em></td>
  </tr>
</table>

### Implementation
#### bicycleGridRiding(gym.Env) -> Cyclists Agent And Environment
- **Attributes**
  - **`nrow`** float, the number of rows in environment. -> grid environment
  - **`ncol`** float, the number of columns in environment. -> grid environment
  - **`moveProb`** n_states * n_actions ndarray, moving probablity of an agent at each state.
  - **`attrTable`** pd.Dataframe, state attributes.
  - **`action_space`** spaces.Discrete, action space.
  - **`observation_space`** spaces.Dict, observation space.
  - **`_actions`** dictionary, possible actions taken by agent.
  - **`defaultReward`** array, default reward at each state.
- **Properties**
  - **`states`**
  - **`coordinate_to_state`**
  - **`state_to_coordinate`**
  - **`state_to_feature`**
  - **`action2Index`**
  - **`index2Action`**
  - **`_get_obs`**
  - **`_get_info`**
  - **`_move`**
  - **`step`**
- **Methods**
  - **`transitFunc`**
  - **`transitFuncArray`**
  - **`rewardFunc`**
  - **`reset`**

## 2. [DPforGridBike_v1.py](https://github.com/XiWen0627/MaxEnIRL_Try/blob/main/IRL/DPforGridBike_v1.py)
## 3. [MEDIRL.ipynb](https://github.com/XiWen0627/MaxEnIRL_Try/blob/main/IRL/MEDIRL.ipynb)
