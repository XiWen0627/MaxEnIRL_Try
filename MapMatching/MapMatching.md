# Map Matching For Dockless-Bike-Sharing Trajectory Data
## Algorithms Implemented
- [Leuven Map Matching](https://leuvenmapmatching.readthedocs.io/en/latest/index.html)
## My Contribution
- The output of the Leuven MapMatching Algorithm is a sequence of road segments, which restricts the ability to fully understand the relationship between individual trajectory points and their mapped results.
- This project further developes the Leuven MapMatching Algorthm to obtain the match result for each trajectory point.
- For more details, please visit my source code and its documentation.
  
## Requirements
- Numpy
- Pandas
- Geopandas
- Leuven MapMatching
- Osmnx (optional)
  
## Documentation
### Data
- DBS Trajectory Data
  - Source from a well-known DBS service provider collected from Nov 1st to Nov 30th, 2017.
  - 
  
- Road Network  
  - Open data from Open Street Map(OSM).
  - Obtain by Python package Osmnx. 
  
### Functions
- `construct_road_network(mapcon, shapefile, coordinate='epsg:4547')`
- `load_road_network(network_dir, map_con)`
