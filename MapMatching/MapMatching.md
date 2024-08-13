# Map Matching For Dockless-Bike-Sharing Trajectory Data
## Algorithms Implemented
- [Leuven Map Matching](https://leuvenmapmatching.readthedocs.io/en/latest/index.html)
## My Contribution
- The output of the Leuven MapMatching Algorithm is a sequence of road segments, which restricts the ability to fully understand the relationship between individual trajectory points and their mapped results.
- This project further developes the Leuven MapMatching Algorthm to obtain the match result for each trajectory point.
  
## Requirements
- Numpy
- Pandas
- Geopandas
- Leuven MapMatching
- Osmnx (optional)
  
## Documentation
### Data
- **DBS Trajectory Data**
  - I’m sorry, but I cannot provide all the DBS trajectory data in this document due to privacy concerns. Instead, I have included a small sample to help you customize your own programs.
  - Source from a well-known DBS service provider collected from Nov 1st to Nov 30th, 2017.
  - [Samples and data formats](https://github.com/XiWen0627/MaxEnIRL_Try/blob/main/MapMatching/Sample.txt).
  
- **Road Network**  
  - Open data from Open Street Map(OSM).
  - Obtain by Python package Osmnx. 
  
### **Functions**
**Attention**: If you're unsure how to construct a class for map matching, please visit [examples](https://leuvenmapmatching.readthedocs.io/en/latest/usage/introduction.html).

#### **mapMatching.py**
- **`construct_road_network(mapcon, shapefile, coordinate='epsg:4547')`** Construct a road network suitable for further operations using the Leuven MapMatching.
- **`load_road_network(network_dir, map_con)`** Load node and link information from existing files.
  
#### **join.py**
Attach atrributes to matched trajectory points.
