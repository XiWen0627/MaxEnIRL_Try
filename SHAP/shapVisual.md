# Exploratory Visualization of SHAP

## Algorithms Implemented
- [SHAP](https://github.com/shap/shap)
    
## My contribution
- Change the style of the Shapley dependence plot without exporting the Shapley values.
- **Cluster the individual Shap values to clarify the characteristics of streetscapes favored by cyclists**(Not successful, but it can be regarded as a potential direction in similar research scenarios).

## Requirements
- Numpy
- Pandas
- Scipy
- Matplotlib
- Seaborn
- SHAP

## Project Documentation
[**1. Explanatory Visualization of Shapley Values**](https://github.com/XiWen0627/MaxEnIRL_Try/blob/main/SHAP/SHAP_visual.ipynb)

<table align="center">
  <tr>
    <td ><img src="https://github.com/user-attachments/assets/8e78ee2d-016c-47d5-bde3-b60c11eb08b9" width="500" alt="Figure 7" /></td>
    <td><img src="https://github.com/user-attachments/assets/25bd3432-4859-4349-a983-0338ff5a276e" width="560" alt="Figure 8" /></td>
  </tr>
  <tr align="center">
    <td><em>Feature SHAP Value of Each Sample</em></td>
    <td><em>Average SHAP Value of Each Feature</em></td>
  </tr>
</table>

<table align="center" width="100%">
  <tr>
    <td ><img src="https://github.com/user-attachments/assets/0909ee47-762c-4f49-94b1-98ceb58de5a0" width="500" alt="Figure 9" /></td>
    <td><img src="https://github.com/user-attachments/assets/d8292243-153d-4ad4-80c6-c655bf5e831a" width="510" alt="Figure 10" /></td>
    <td><img src="https://github.com/user-attachments/assets/eceaa21a-5aff-4718-b5e3-944a4b115826" width="500" alt="Figure 11" /></td>
  </tr>
  <tr align="center">
    <td><em> Green View Index SHAP</em></td>
    <td><em>Sky View Factor SHAP</em></td>
    <td><em>Motorcycle SHAP</em></td>
  </tr>
</table>

<table align="center" width="100%">
  <tr>
    <td ><img src="https://github.com/user-attachments/assets/8afe615c-77ee-480b-970e-b11d08e4b168" width="800" alt="Figure 12" /></td>
    <td><img src="https://github.com/user-attachments/assets/315bde81-dd2b-44a8-8e02-0648614bb61d" width="400" alt="Figure 13" /></td>
  </tr>
  <tr align="center">
    <td><em>SHAP Interaction Plot</em></td>
    <td><em>SHAP Interaction Value Matrix</em></td>
  </tr>
</table>

<table align="center" width="100%">
  <tr>
    <td ><img src="https://github.com/user-attachments/assets/cde6f2c9-4cf3-4a49-9bcc-e8b1c9ccf11e" width="500" alt="Figure 12" /></td>
    <td><img src="https://github.com/user-attachments/assets/60f84dff-a9c7-4caa-971e-d506f89ac1cf" width="500" alt="Figure 13" /></td>
    <td><img src="https://github.com/user-attachments/assets/775b32cd-bd7c-4e90-b51b-8cc14cc286d0" width="500" alt="Figure 14" /></td>
  </tr>
  <tr align="center">
    <td><em>Interaction between GVI & Wall</em></td>
    <td><em>Interaction between Motorcycle & SVF</em></td>
    <td><em>Interaction between Motorcycle & Wall</em></td>
  </tr>
</table>

[**2. Parallel Coordinate Plot of Shapley Values**](https://github.com/XiWen0627/MaxEnIRL_Try/blob/main/SHAP/ParallelCoordinatePlot.ipynb)

<table align="center" width="100%">
  <tr>
    <td ><img src="https://github.com/user-attachments/assets/120680bb-3ce8-494f-8fad-409f3082bac3" width="100%" /></td>
  </tr>
  <tr align="center">
    <td><em>Parallel Coordinate Plot</em></td>
  </tr>
</table>

[**3. Cluster Analysis of Shapley Values**](https://github.com/XiWen0627/MaxEnIRL_Try/blob/main/SHAP/Cluster.ipynb)

The parallel coordinate plot reveals specific patterns in the distribution of Shapley values.
As a result, we employ several clustering methods to uncover preference patterns from shapley values.

<table align="center" width="100%">
  <tr>
    <td ><img src="https://github.com/user-attachments/assets/72126989-3aa0-417c-a98d-555297180f92" width="100%" /></td>
  </tr>
  <tr align="center">
    <td><em>Stem Plot</em></td>
  </tr>
</table>

<table align="center" width="100%">
  <tr>
    <td ><img src="https://github.com/user-attachments/assets/92056a91-d81c-44d6-9f94-a73b9f25d2e4" width="1200" /></td>
  </tr>
  <tr align="center">
    <td><em>Principal Component Analysis of SHAP</em></td>
  </tr>
</table>

<table align="center" width="100%">
  <tr>
    <td ><img src="https://github.com/user-attachments/assets/3fe5d4b4-9b60-4041-b490-905a28151772" width="100%" /></td>
  </tr>
  <tr align="center">
    <td><em>Hierachy Clustering Based on PCA</em></td>
  </tr>
</table>


<table align="center" width="100%">
  <tr>
    <td ><img src="https://github.com/user-attachments/assets/c237d035-212b-466f-afa6-ec73814a0bff" width="1200" /></td>
  </tr>
  <tr align="center">
    <td><em>Streetscape Characteristic in Cluster1</em></td>
  </tr>
</table>

<table align="center" width="100%">
  <tr>
    <td ><img src="https://github.com/user-attachments/assets/52e31a5f-d340-4fae-8bf1-4c84257496d6" width="1200" /></td>
  </tr>
  <tr align="center">
    <td><em>Streetscape Characteristic in Cluster2</em></td>
  </tr>
</table>
