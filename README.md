# The Application of Maximum Entropy Deep Inverse Reinforcement Learning in Cycling.
Discovering Cyclists' Street Visual Preferences Through Multi-Source Big Data Using Deep Inverse Reinforcement Learning
# Part 1. Introduction to My Research Scenario
Our study proposes a novel framework aimed to **automatically quantify and interpret cyclists’ complicated street visual preferences from cycling records by leveraging maximum entropy deep inverse reinforcement learning(MEDIRL) and explainable artificial intelligence(XAI).** Specifically, we adapt MEDIRL model for efficient estimation of cycling reward function by integrating DBS trajectory and SVIs, which serves as a representation of cyclists’ preferences for street visual environments during routing.  

Implemented in Bantian Sub-district, Shenzhen, our framework demonstrates the feasibility and reliability of MEDIRL in discovering cyclists’ street visual preferences. Detailed analysis indicates the complex relationships between cyclists’ visual preferences and street elements. Our proposed framework not only advances the understanding of individual cycling behaviors but also provides actionable insights for urban planners to design bicycle-friendly streetscapes that prioritize cyclists’ preferences and safety.

## Concept Clarification
**`Cyclists' Route Choice`** A continuous road selection procedure, constrained by origin and destination (OD) and influenced by the built environment, represents how cyclists navigate from place to place.

**`Cyclists' Preferences`** General patterns identified from individual route choice process.

## Research Topic
- How to discover cyclists' prefernces from their routing process?  

- More specifically, we aim to discover cyclists’ general **street visual preferences** based on their continuous route decision procedures influenced by streetscape characteristics.

## Our Solution
we propose an **Inverse Reinforcement Learning(IRL) based framework** to **quantify** and **interpret** cyclists' visual preferences along urban streets, focusing specifically on their cycling process. The overall research idea is described as follows. 

- **Data**: Dockless-Bike-Sharing(DBS) Trajectory Data & Street View Imagery(SVIs) 
- **Methodology**: Maximum Entropy Deep Inverse Reinforcement Learning(MEDIRL) ＋ Explainable Artificial Intelligence(XAI)  

<p align="center">
  <img src="https://github.com/user-attachments/assets/19a350ce-f5fa-4df5-b349-cb7d895330ab"  title="Figure 1. Research Conceptual Framework" />
  <br />
  <em>Figure 1. Research Conceptual Framework</em>
</p>

  
### Fitness of reseach question and methodology
- IRL is effective in mining sequential dependencies and semantic information in trajectory data.
- IRL is flexible in integrating Deep Learning(DL) architectures and high dimensional features, which helps capture thenon-linear and complicated nature of cycling preferences.
- The unique training process of IRL makes it more behaviorally interpretable compared to conventional DL methods and facilitates further simulation and optimization.  

### Framework
The overall workflow comprises three distinct steps, as illustrated in **Figure 2**. Practically, cycling is treated as a route decision process constrained by road spatial networks, taking into account origin-destination (OD) pairs and street visual environments. 

- Formalize cycling process as a Markov Decision Process (MDP) by integrating SVIs and DBS trajectories to **detail cycling procedures outlined earlier** for further analysis.
- Employ IRL to **recover the underlying reward function of MDP from observed trajectory data**. The reward function reflects the general principal cyclists follow, influenced by environmental factors, and served as quantified street visual preferences.
   - Approximate this reward function using a combination of maximum entropy model and deep neural network (DNN) to balance diverse cyclist preferences and capture their non-linear nature (MEDIRL). 
   - Validate learned results by comparing similarities between reconstructed and real trajectories.
-  Utilize XAI to **interpret the contributions** of specific visual elements to cyclists' street visual preferences.  

<p align="center">
  <img src="https://github.com/user-attachments/assets/e83100dc-7758-4f08-b447-a266d79c6dda"  title="Figure 2. Overall Workflow" />
  <br />
  <em>Figure 2. Overall Workflow</em>
</p>

## Partial Results of Our study
### Desciptive Statistics of Model Output
#### Model Settings
- **Quantify cyclists’ general street visual preferences** by training the MEDIRL model in Bantian Sub-district in Longgang District, Shenzhen.
- Model cyclying process as an MDP with a 100m grid size and a 0.99 discount rate.
- Approximate the relationship between state features and the reward function utilizing an Multi-Layer Perceptron (MLP) with 4 hidden layers and Rectified Linear Units (ReLU). Notably, **the value of reward function at each state represents quantified cycling preferences**.
- Discover the spatial dependence and spill-over effects of cyclists' preferences by utilizing Local Indicators of Spatial Association (LISA).
    
#### Model Result
- Spatial distribution of normalized cycling preferences.
- Spatial clustering of normalized cycling preferences.
    
<table align="center" width="100%">
  <tr>
    <td><img src="https://github.com/user-attachments/assets/14d4fda5-8b67-4c46-8bff-83952108d90e" width="500" alt="Figure 3" /></td>
    <td><img src="https://github.com/user-attachments/assets/c0437627-a85a-485c-af7d-56efe4aa7808" width="500" alt="Figure 4" /></td>
  </tr>
  <tr align="center">
    <td><em>Figure 3. Spatial Distribution of Normalized Reward</em></td>
    <td><em>Figure 4. Spatial Clustering of Normalized Reward</em></td>
  </tr>
</table>

### Model Evaluation
#### Model Settings
- Employ agent-based model where the output of trained MEDIRL model serves as behavioral rules to generate new trajectories.
- Measure the **similarity between the state occurrence frequencies** in two trajectories using **Jensen-Shannon Divergence(JSD)**.
- Assess the **similarity between individual trajectories** utilzing **Common Part of Commuters(CPC)**.
  
#### Model Results
- Statistical similarity between actual and synthetic trajectories.
- Similarity between actual and synthetic trajectories at the individual level.

<table align="center" width="100%">
  <tr>
    <td ><img src="https://github.com/user-attachments/assets/9c0c3ce0-f77f-482d-b53c-5e7d0e8a30e7" width="1000" alt="Figure 5" /></td>
  </tr>
  <tr align="center">
    <td><em>Figure 5. SVF Distributions of Real and Synthetic Trajectories</em></td>
  </tr>
</table>

<table align="center" width="100%">
  <tr>
    <td ><img src="https://github.com/user-attachments/assets/a906abf4-42f0-4dc2-88db-588da60b33e8" width="250" /></td>
    <td ><img src="https://github.com/user-attachments/assets/d4d0b14b-8591-4209-8fcb-3eb140efd66a" width="250" /></td>
    <td ><img src="https://github.com/user-attachments/assets/8dcbbf29-9a9f-423c-a24a-b635fac3156b" width="250" /></td>
    <td ><img src="https://github.com/user-attachments/assets/b3443228-44a1-4825-8b0c-33e2cacfc490" width="250" /></td>
  </tr> 
</table>

<p align="center">
  <em>Figure 6. Selected Trajectories and Corresponding Generated Data</em>
</p>

### Interpretability of Environmental Preference of Route Decision Process
#### Importance of the Street Visual Elements

<table align="center">
  <tr>
    <td ><img src="https://github.com/user-attachments/assets/8e78ee2d-016c-47d5-bde3-b60c11eb08b9" width="500" alt="Figure 7" /></td>
    <td><img src="https://github.com/user-attachments/assets/25bd3432-4859-4349-a983-0338ff5a276e" width="560" alt="Figure 8" /></td>
  </tr>
  <tr align="center">
    <td><em>Figure 7. Feature SHAP Value of Each Sample</em></td>
    <td><em>Figure 8. Average SHAP Value of Each Feature</em></td>
  </tr>
</table>

#### Nonlinear and Threshold Effect of Each Street Visual Elements on Cycling Reward
  
<table align="center" width="100%">
  <tr>
    <td ><img src="https://github.com/user-attachments/assets/0909ee47-762c-4f49-94b1-98ceb58de5a0" width="500" alt="Figure 9" /></td>
    <td><img src="https://github.com/user-attachments/assets/d8292243-153d-4ad4-80c6-c655bf5e831a" width="510" alt="Figure 10" /></td>
    <td><img src="https://github.com/user-attachments/assets/eceaa21a-5aff-4718-b5e3-944a4b115826" width="500" alt="Figure 11" /></td>
  </tr>
  <tr align="center">
    <td><em>Figure 9. Green View Index SHAP</em></td>
    <td><em>Figure 10. Sky View Factor SHAP</em></td>
    <td><em>Figure 11. Motorcycle SHAP</em></td>
  </tr>
</table>

#### Interactions Between Key Street Visual Elements

<table align="center" width="100%">
  <tr>
    <td ><img src="https://github.com/user-attachments/assets/cde6f2c9-4cf3-4a49-9bcc-e8b1c9ccf11e" width="500" alt="Figure 12" /></td>
    <td><img src="https://github.com/user-attachments/assets/60f84dff-a9c7-4caa-971e-d506f89ac1cf" width="500" alt="Figure 13" /></td>
    <td><img src="https://github.com/user-attachments/assets/775b32cd-bd7c-4e90-b51b-8cc14cc286d0" width="500" alt="Figure 14" /></td>
  </tr>
  <tr align="center">
    <td><em>Figure 12. Interaction between GVI & Wall</em></td>
    <td><em>Figure 13. Interaction between Motorcycle & SVF</em></td>
    <td><em>Figure 14. Interaction between Motorcycle & Wall</em></td>
  </tr>
</table>


# Part 2. Documentation
## 1. Map Matching
### Algorithms Implemented
- [Leuven MapMatching](https://leuvenmapmatching.readthedocs.io/en/latest/)
### My Contribution 
- The output of the Leuven MapMatching Algorithm is a sequence of road segments, which restricts the ability to fully understand the relationship between individual trajectory points and their mapped results.
- This project further developes the Leuven MapMatching Algorthm to obtain the match result for each trajectory point.
- For more details, please visist my source code and its documentation.

## 2. Semantic Segmantation
### Algorithms Implemented

## Inverse Reinforcement Learning
### Algorithms Implemented
- Linear programming IRL. From Ng & Russell, 2000. Small state space and large state space linear programming IRL.
- Maximum entropy IRL. From Ziebart et al., 2008.
- Deep maximum entropy IRL. From Wulfmeier et al., 2015; original derivation.
- [Inverse-Reinforcement-Learning From Mattew Alger et al., 2017.](https://github.com/MatthewJA/Inverse-Reinforcement-Learning?tab=readme-ov-file)
