# MaxEnDeepIRL Application for uncovering visual preference in cycling route decision procedure.
XiWen0627's Master Project : Unraveling  Built Environment Preference to Route Choice via IRL Approach 

## Introduction: importance of cycling and DBS system 
Cycling, widely recognized as a sustainable means of transportation, promotes outdoor activities and presents a potential solution to urban challenges such as traffic congestion and air pollution. The emergence of Dockless Bike Sharing (DBS) has further enhanced cycling by significantly improving accessibility to active travel. Consequently, DBS has garnered widespread adoption across numerous countries and has surged in popularity, particularly following the Covid-19 pandemic. Unlike motor vehicle travel, cyclists make decisions based not only on long-term travel plans but also on immediate environmental factors. However, urban planning often fails to accommodate the unique characteristics of cycling behavior, resulting in insufficient support for cyclists. Therefore, gaining a detailed understanding of cyclists' preferences and behaviors within DBS systems is crucial for bicycle-friendly urban planning.     **->Background**

The integration of GPS devices in DBS systems generates valueable location-based data, sparking significant academic interest in the fields of transportation and urban planning. Researchers have diligently studied cyclists' behavior and its correlation with various environments using data driven approaches. However, existing studies often prioritize analyzing cyclists' Origins and Destinations (ODs) rather than examining the detailed cycling procedures. This oversight may limit urban planners' understanding of how cyclists' behaviors relate to their on-site environments, which are crucial for understanding environmental preferences and influencing the scientific design of urban streets. To tackle the oversight of cyclists' preferences regarding their surroundings during DBS cycling, we have identified two key challenges.    **The importance of cycling procedure**

**The primary challenge lies in comprehensively understanding and describing the detailed procedures of cycling**. One of the most common and intuitive methods is Route Choice Modeling(RCM), which predicts the possible paths that individuals follow during their trips. Typically, the principles uncovered in route selection are often interpreted as cycling preferences. The RCM framework is highly regarded for its potential and effectiveness in integrating new data and methodologies. Some studies integrate DBS trajectory data into RCM, treating cycling volume as a proxy for subjective preferences. However, DBS volume, affected by supply, demand, and temporal patterns, maks it difficult to accurately infer cyclists’ preferences from aggregated data. Some researchers have combined Agent-Based Model (ABM) with RCM to simulate real-time interactions from cyclists’ perspective. Yet, the development of such studies often rely on pre-defined indices and behavioral rules,  potentially overlooking implicit information in cyclists' decision-making processes. In other domains like ridesharing, deep learning methods like LSTM and GRU capture spatiotemporal dependencies in trajectories. However, cyclists, who both use and navigate the environment, exhibit different decision-making patterns compared to taxi passengers. Thus, understanding individual cycling processes is crucial for data-driven approaches to uncover cyclists' preferences.

**Another challenge followed immediately is to capture and uncover cyclists' environmental preferences in cycling procedures.** Traditionally, traffic surveys are employed to explore the relationship between cycling behavior and macro-level Built Environment (BE) factors like infrastructure and land use. Yet, they often overlook preferences for various micro-level Street Environment (SE) visual features, limiting practical applications in urban planning. Moreover, existing studies frequently assume linear utility functions, which fail to adequately capture the complex and nonlinear routing decisions influenced by environmental factors. The emergence of multisource big data and Artificial Intelligence (AI) offers new opportunities to accurately capture cyclists' experiences in urban environments. Street View Images (SVI) and DBS trajectory data allow for a human-centric characterization of cyclists' routing decisions, with AI advancements extracting intricate preferences from diverse data sources. However, deep learning methods often lack explanatory, resulting in limited applicability in urban planning. In summary, these data-driven technological advances pave the way for a cognitively interpretable model framework to uncover cyclists' environmental preferences from their cycling procedures.

There are two gaps in the data-driven research on the cyclists' preferences for street environments. The first is how to understand cycling procedures while considering the impact of SE. The second lies on how to extract coherent cycling preferences towards SE from highly random and heterogeneous big data.

In recent years, reinforcement learning(RL) has shown excellent ability in dealing with sequential decision-making problems in random situations (Li, 2023; Qin et al., 2022), and has been widely adopted in trajectory data mining, route recommendation, vehicle repositioning and other researches. However, RL methods often rely on predefined operating rules (Rong et al., 2016; Yu et al., 2019; Yu & Gao, 2022), while the Inverse Reinforcement Learning(IRL) reverses the above process and recovers the principles that the agent potentially follows from the cycling information through the combination of simulation and training(Abbeel & Ng, 2004; Ng & Russell, 2000), which provides an opportunity for researchers to measure cycling procedures and reveal the complex cyclists' environmental preferences from given data.

To build upon existing research and bridge identified gaps, we propose a framework that integrates multi-source big data and IRL to quantify and interpret the preferences in the route choice process. Specifically, we take Bantian Community, Longgang District, Shenzhen as the empirical research area, and on the basis of reconstructing the route decision-making process of cyclists, we integrate data such as DBS trajectories and street view images (SVI), and employ the maximum entropy deep inverse reinforcement learning (MEDIRL) method to automatically reveal the position preference from the perspective of cyclists. To verify the method's effectiveness, we simulate trajectories based on the extracted position preferences, and assess the similarities between these simulated trajectories and the real trajectories. Finally, we adopt explainable artificial intelligence (XAI) techniques to illustrate the complex relationship between cyclists' real-time perception and route decisions.
The major contributions of this study can be concluded in three aspects:

 1. We proposed a comprehensive framework for unraveling the cycling preference of urban streets, emphasizing a detailed process that goes beyond conventional OD analysis. DBS trajectories offer insights into the entire cycling process, enabling a more nuanced understanding of cycling route decision, while SVI provides opportunities to accurately describe street-level visual features related to cycling trips.
     
 2. Our study proposed a novel cycling preference quantification and interpretation method, leveraging results derived from MEDIRL and XAI models. This method automatically establishes a robust link between the street environment and cycling behavior, explicitly considering their complex interactions. Compared to existing methods, our data-driven method offers enhanced reliability and reasonability.
    
 3. We applied our proposed framework in a real-world scenario, specifically in Ban Tian community in Longgang District, Shenzhen. The quantification and interpretation of cyclists’ preferences were conducted by considering both path-level (i.e., OD) and link-level characteristic for each trajectory. The practical case study underscores the feasibility and reliability of our proposed method.

The rest of the paper is organized as follows. Section 2 introduces the study area and dataset. Section 3 outlines our methodological framework, including maximum entropy deep inverse reinforcement learning, explainable artificial intelligence, and the metrics we utilized. In section 4, we conduct an empirical research in study area to validate the reliability and explainability of our approach. Section 5 summarizes this research and outlines the future work.


## Study Area and Dataset
### Study area
In our study, we focus on Bantian Sub-district in Longgang District, Shenzhen as the research area. This area is characterized by relatively consistent cycling demand, convenient traffic facilities, comfortable travel conditions and diverse environmental attributes, making is suitable for investigating cyclists' preferces to environmental elements.

Bantian Sub-district has served as a showcase for the integration of technology and urban development within the Guangdong-Hong Kong-Macao Greater Bay Area. It is particularly renowned for its technology industry, and attracts a large number of young commuters. Situated on the border of three districts in Shenzhen, the area offers diverse transportation options for residents. Specifically, it boasts well-established public transportation, including 2 metro lines with 22 subway stations, complemented by a sufficient supply of DBS services. The total road network spans 38.97 km, comprising 5,229 intersections and 7,039 road segments accessible to cyclists. Moreover, Bantian Sub-district provides favorable conditions for year-round outdoor cycling with an average temperature of 23.3°C and gentle slopes averaging less than 10°. Its spatial heterogeneity in natural and socio-economic environments enables us to distinguish between various areas and extract typical attributes.
  
### Data
#### DBS Trajectory Data
The DBS trajectory data utilized in this study were sourced from a well-known DBS service provider, comprising **10,00** independent trips collected in Bantian Sub-district from Nov 1st to Nov 30 th, 2017. Specifically, our DBS trajectory data encompass basic order details such as Order ID,  User ID, DBS ID, Start Time, End Time, Start Coordinate, End Coordinate and a sequence of trajectory points collected at three-second interval, as illustrated in Table 1. Preproccessing of DBS trajectory data involves three stages: raw data cleaning, map matching and data filtering.

First of all, we identified and eliminated the counter-intuitive cycling trips in order to enhance data quility. Specifically, we filtered out trips less than three minutes or exceeding one hour, considering the rebalancing process. Additionally, we removed the trajectory points where speeds exceeded 30 km/h, likely due to poor GPS signal. Following this data cleaning process, we obtained a dataset of **10,000** distinct cycling trips suitable for map matching.

Second, we mapped trajectory points onto the road network using a Hidden Markov Model(HMM) to better approximate the link-based decision-making process of DBS cycling. This method successfully assigned a sequence of road segments to the trajectory points. Furthermore, to validate the effectiveness of map-matching process, we randomly selected several mapped trajectories and compared them with raw data. As shown in **Figure 1**, this comparison demonstrated convincing result.

Finally, we filtered the mapped DBS trajectory data using specific criteria informed by preliminary literature reviews and exploratory analysis. Given the subtropical location and an average November temperature of 19.7℃, our data comprehensively represented cycling behavior. To account for the direct impact of weather conditions on cycling, we excluded data from two rainy days(Nov 9th and Nov 16th). Additionally, to mitigate potential biases in estimating preferences due to temporal changes, we focused our analysis on cyclists' behaviors during weekdays and daylight hours. Furthermore, our DBS trajectories deviated significantly from the shortest path when applying the Dijkstra Algorithm, aligning with findings in current literature. Specifically, approximately 83.8% of cyclists selected the shortest path for trips involving fewer than five road segments. Therefore, we concentrated on cycling trips with more than five road segments, resulting in **10000** distinct trip trajectories as the valid dataset for further investigation. The results of exploratory data analysis are shown in Appendix 1.

#### Street View Images
Street View Images (SVI), a widely used type of big geospatial data, offers detailed visual representations of urban physical environments. Additionally, SVI implicitly capture invisible information like socio-economic environments and human activities. In our research, SVI remains largely unchanged due to the subtropical climate, making it suitable for simulating cyclists' real-time visual perceptions on urban streets.

The SVI data used in this study were obtained from Baidu, comprising 7924 distinct panoramas in Bantian Sub-district, primarily captured in 2017. Semantic segmentation was conducted using the DeepLabV3+ model pretrained on the Cityscapes dataset, to extract visual elements. As a result, we identified eight primary categories and 21 detailed subcategories of street scenery, as detailed in Table 2. Subsequently, we quantified the cyclists' visual perceptions by calculating the ratios of the pixels assigned to each category relative to the total number of pixels in the image, normalized using min-max scaling between zero and one. The segmentation achieved an average coverage of 97.88%, indicating its applicability for further investigation.

## Methodology
### Framework
The detailed procedure of cycling  involves abundant temporal and semantic information, providing researchers with insights to cyclists' behavioral patterns. However, the complexity of cycling environments and the stochastic nature of behaviors present challenges in uncovering cyclists' preferences from  detailed procedures.

In our study, we propose a data-driven framework designed to quantify and interpret cyclists' visual preferences along urban streets, focusing specifically on their cycling procedures. The overall workflow comprises three distinct steps, as illustrated in **Fig 1**. cycling is treated as a route decision process constrained by road spatial networks, taking into account origin-destination (OD) pairs and street visual environments. We formalize it as a Markov Decision Process (MDP) integrating SVI and DBS trajectories. Secondly, we employ Maximum Entropy Deep Reinforcement Learning (MEDIRL) to quantify cyclists’ environmental preferences derived from the cycling procedures outlined earlier. We validate learned results by comparing similarities between reconstructed and real trajectories. Finally, we utilize explainable Artificial Intelligence(XAI) to interpret the contributions of specific visual elements to cyclists' environmental preferences.

![methodologyFramework](https://github.com/user-attachments/assets/e83100dc-7758-4f08-b447-a266d79c6dda)

### Preliminaries and Problem Formulation
Cycling procedures can be regarded as a MDP, which provides a general framework for modeling the sequential decision process of a cyclist. A MDP is generally defined as $M=\{S,A,T,R,γ\}$, where $S$ denotes the state space, representing the set of possible positions the agent can be; A denotes the set of possible actions the agent can take. Generally speaking, the sequence of state-action pairs is also referred to as the trajectory, i.e., \{(s_1,a_1 ),(s_2,a_2 ),···,(s_t,a_t )\}. T(s_t,a_t,s_(t+1) ) denotes a transition model that determines the next state s_(t+1) given the current state s_t and action a_t. R(s,a) is the reward function, defined as the feedback obtained by the agent when taking action a∈A in state s∈S. In the modeling of sequential decision-making problems, the policy π describes the moving strategy at each stat. The agent's policy is often non-deterministic, and this stochastic policy can be intuitively understood as the probability of the agent taking action a_t∈A given the current state s_t∈S, denoted as Pr(a_t│s_t ). The quality of a policy is often evaluated based on its long-term return, and the most popular definition is the discounted return, i.e., G_t= ∑_(i=0)^(+∞)▒〖γ^i r_(t+1) 〗  γ∈[0,1]. When γ=0, the agent is short-sighted, only focusing on immediate rewards and disregarding the temporal dependencies of the policy; as γ approaches 1, the agent considers future rewards more. In the RL setting, the reward function R(s,a)  s∈S,a∈A is given, and the objective is to find the optimal policy π^* that maximizes the expected cumulative reward for the agent. This is accomplished by sampling trajectory data through continuous interaction between the agent and the environment.
In our study, cyclists are treated as agents in an MDP. Solving this MDP model will give us the optimal decision strategy for each different location. Specifically, we can define the elements of the MDP as follows: 
 
- **State**: Each state s∈S is a vector used to describe the basis of a cyclist’s decision-making. Due to the simulation-based learning of cyclists behavioral rules under the given ODs, the agent cannot obtain specific path information beforehand. Therefore, in our study, each state is a link in road network indicating the current location and corresponding perceptions of the cyclist, denoted as s=\{Dest,Pos,BE\}. The vector Dest is the final state of current agent. The vector Pos is utilized to represented the links and the 100m grid units where agent is located, serving as the core determinants of its location. Depending on data availability, the real-time perception of the cyclist can be incorporated, but is outside the scope of this study. Alternatively, we will regard SVI as great approximation of cyclist’s perception, following most prior works(Zhang et al., 2018). As a result, the vector BE is composed of the semantic elements of SVI. (The relationship between SVI & perception)
- **Action**: An action a∈A indicates the grid-to-grid movement choice under the restriction of road networks. Liang & Zhao (2022) has shown that a directional representation can yield better route prediction performance, as a result, we define a global action space A consisting of 9 movement directions — forward (F), forward left (FL), left (L), backward left (BL), backward (B), backward right (BR), right (R), forward right (FR), and stay(ST), as shown in Figure.7. Note that, although these 9 directions represent a comprehensive set of all potential actions to be taken anywhere, only a subset of them are applicable for most states. In order to account for the specific layout of the local network, we have also defined a local action space A_s∈A to capture all valid actions at each state s. 
- **Policy**: Generally, a policy, denoted as π(a|s), dictates the actions an agent takes in each state s∈S. These policies can vary, guiding agents to select different actions even in similar states, resulting in diverse trajectories. In our study, the policy describes how cyclists make route decisions, namely the process of selecting links under the influence of the built environment. The optimal policy, denoted as π^*, represents the most representative route decision-making pattern for cyclists.
- **Reward function**: The reward function R(s,a) characterizes cyclists' preferences for the built environment in the route decision-making process. Corresponding to the optimal policy π^*, R^* represents the reward function that best explains cyclists' route preferences. Additionally, in this study, a set of parameters θ is used to fit the utility function of cyclists interacting with complex and heterogeneous environments, thereby replacing the location-based reward representation method to store the mapping between states, actions, and their rewards, denoted as R_θ (s,a)=θ(Pos,BE,dest). As mentioned earlier, the rewards dictate the actions of the agent. Therefore, compared to the agent's policy, the reward function R is more likely to provide researchers with insights into its decision-making mechanism.

In summary, our study models the cycling procedure as an MDP. It can be described as the process where cyclists make continuous decisions about street selection based on their immediate perceptions, aiming to maximize their cumulative return. However, it is often difficult to characterize the relationship between cyclists' immediate perceptions and their decision-making tendencies using a predefined reward function, which significantly restricts the applicability of such approaches. Fortunately,  Andrew Ng's framework of Inverse Reinforcement Learning (IRL) (Abbeel & Ng, 2004; Ng & Russell, 2000) offers a solution. By reversing the RL process, IRL extracts the reward function from demonstrated data. This methodological approach provides a solid foundation for addressing the challenges mentioned earlier.

## Experiments
To quantify the cyclists' environmental preferences, we evaluate our proposed MEDIRL model by comparing actual and predicted route choices. The prediction resembles an agent-based simulation where the output of trained MEDIRL model serves as behavioral rules to generate new trajectories. Subsequently, we evaluate similarity between synthetic and real trajectories based on statistical and route characteristics.

Furthermore, to interpret the proposed model, we employ the SHAP algorithm to examine the contribution of each street visual element within each unit, thereby assessing the importance of each feature and their complex relationships in cyclists' preferences thatwe extracted before.

### Model Evaluation
We train the MEDIRL model on integrated SVI and DBS trajectories to automatically extract cyclists' environmental preferences. Specifically, our approach employ a Multi-Layer Perceptron (MLP) with 4 hidden layers and rectified linear units (RELU) to approximate the relationship between state feature representations and rewards. Following established methods in the field, we evaluate our model by comparing the similarity between actual cyclists' trajectories and synthetic data generated using the learned preferences.

We initially measure the similarity between real and synthetic trajectories based on their statistical characteristics as summarized in Fig.9. In our dataset, the calculated JSD between the distributions is 0.3484, indicating a significant degree of resemblance between the synthetic and real trajectories in terms of fluctuation ranges. However, we observed that the synthetic trajectories exhibit higher means and smaller variances compared to the real trajectories. This suggests that further refinement of the reward function is necessary to better capture the characteristics of longer trips. 

Moreover, we utilize the Sørensen-Dice coefficient to evaluate the similarity between individual trajectories. The probability density function is illustrated in **Figure.10 (a)**. Our findings reveal that, on average, synthetic trajectories overlap with real ones by 66.67% for each OD pair. Our descriptive statistical analysis, as shown in **Figure 10**, provides deeper insights into their similarity based on decision frequency. Our results highlight that trajectories generated under learned reward excel in reflecting preferences for medium to short-distance cycling paths. Employing the elbow method to analyze the relationship between decision frequency and CPC, we observed that that once the number of decisions exceeds 18 in a single trip, the rapid decline in similarity between trajectories halts. 

We also visualize the spatial distribution of cycling reward for each state, as shown in Figure X. To gain more intuition about model outputs,  a visualization example of a real trajectory from the data and its corresponding synthetic trajectory generated by our model is shown in **Figure 4** as a comparison. The results indicate that our link-based MEDIRL model effectively captures cyclists' sequential decision-making patterns from real trajectory data.


## Appendix 1. Exploratory Data Analysis for DBS Trajectory Data
## Appendix 2.
