# MaxEnDeepIRL Application for uncovering visual preference in cycling route decision procedure.
XiWen0627's Master Project : Unraveling  Built Environment Preference to Route Choice via IRL Approach 

## Introduction: importance of cycling and DBS system 
Cycling, widely recognized as a sustainable means of transportation, promotes outdoor activities and presents a potential solution to urban challenges such as traffic congestion and air pollution. The emergence of Dockless Bike Sharing (DBS) has further enhanced cycling by significantly improving accessibility to active travel. Consequently, DBS has garnered widespread adoption across numerous countries and has surged in popularity, particularly following the Covid-19 pandemic. Unlike motor vehicle travel, cyclists make decisions based not only on long-term travel plans but also on immediate environmental factors. However, urban planning often fails to accommodate the unique characteristics of cycling behavior, resulting in insufficient support for cyclists. Therefore, gaining a detailed understanding of cyclists' preferences and behaviors within DBS systems is crucial for bicycle-friendly urban planning.

The integration of GPS devices in DBS systems generates valueable location-based data, sparking significant academic interest in the fields of transportation and urban planning. Researchers have diligently studied cyclists' behavior and its correlation with various environments using data driven approaches. However, existing studies often prioritize analyzing cyclists' Origins and Destinations (ODs) rather than examining the detailed cycling procedures. This oversight may limit our understanding of how cyclists' behaviors relate to their in-situ environments, which are often seen as environmental preferences. Consequently, we identified two key challenges concerning cyclists' preferences for their surroundings during DBS cycling.

**The primary challenge is to understand the detailed procedure of cycling**. 
In contrast to motor vehicle users, cyclists not only make long-term decisions while considering the origins and destinations(OD), but also make immediate choices based on their real-time perceptions. 

Therefore, another challenge 


1.考虑到骑行过程的前提下，从骑行者的实际选择中提取并量化影响关系
popularity among local residents.

The current problem existing in cycling infrastructure and its relationship with street visual elements.
**However, the boom of cycling also poses higher demands on cycling infrastructure development: the facilities often fails to meet the needs of large-scale and diversified cycling, and urgently requiring improvement through planning to enhance cyclists' experiences. Scientific updating and construction of cycling infrastructure relies on the understanding of the cycling behaviors by urban planners.** 
In contrast to motor vehicle users, cyclists not only make long-term decisions while considering the origins and destinations(OD), but also make immediate choices based on environmental factors. Therefore, it is necessary to delve into the relationship between cycling behaviors and built environment.


The second paragraph intend to briefly introduce the importance of cycling procedure

Contributions:
  1. We proposed a comprehensive framework for **unraveing the Location Preference from real trajectory data**
  2.  unravleing the **route-decision preference of urban street**, emphasing a detailed procedure that goes beyond convenlutional OD analysis.



The major contributions of this study can be concluded in three aspects:

- The first gap may be located on the cycling procedure, especially the interaction between cyclists and environments.
- The second gap may be located on the Quatification & Interpretation the interaction from ral trajectory.
  
1. We proposed a comprehensive framework for **unraveling the route-decision preference of urban streets**, emphasizing a detailed process that goes beyond conventional OD analysis. *DBS trajectories offer insights into the entire cycling process, enabling a more nuanced understanding of cycling route decision(Need to be Correct)*. -> The emphasizing of cycling procedure
2. Our study proposed **a novel cycling preference quantification and interpretation method**, leveraging results derived from MEDIRL and XAI models. This method automatically establishes a robust link between the street environment and cycling behavior, explicitly considering their complex interactions. Compared to existing methods, our data-driven method offers enhanced reliability and reasonability. -> Data driven approach
3. We applied our proposed framework in a real-world scenario, specifically in Ban Tian community in Longgang District, Shenzhen. The quantification and interpretation of cyclists’ preferences were conducted by considering both path-level (i.e., OD) and link-level characteristic for each trajectory. The practical case study underscores the feasibility and reliability of our proposed method.
 
The rest of the paper is organized as follows. Section 2 introduces the study area and dataset. Section 3 outlines our methodological framework, including maximum entropy deep inverse reinforcement learning, explainable artificial intelligence, and the metrics we utilized. In section 4, we conduct an empirical research in study area to validate the reliability and explainability of our approach. Section 5 summarizes this research and outlines the future work.
