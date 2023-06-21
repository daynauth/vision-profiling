# NS Visualizer
A visualization tool that takes parsed neuro-network info as input, and generates DOT code that can visualize the model via a DOT visualizer (e.g. https://dreampuf.github.io/GraphvizOnline/#).

An example of the resulting graph can be found in the `example` directory. 
### How to use
```shell
python3 visualizer.py prof.csv dep.csv optional_file_suffix
```
Where prof.csv contains the following profiling info: 
- Layer name
- Average time consumption (in second)
- Output size (in MB)
- Average memory consumption

and dep.csv contains the dependency:
- Source 
- Destination

This visualizer generates a **result_DOT_code_\<suffix>.txt**.

### Notes for input
- In prof.csv, measure the data for every layer only once.
- In dep.csv, measure the dependency for every output, but only before it is used in another layer.
- When recording the dependency, remember to include concat operations.

### Plot the models in NS paper (yolov4, yolox, yolor, faster-rcnn)
The profile and dependency files for those models are stored in the `paper` directory. To generate DOT code for those models, copy the corresponding csv files to the root directory of this project, and run the visualizer. 

---
# NS Colorer
A coloring tool that visualizes the partition of a neuro-network by different colors. Please use a DOT visualizer as above-mentioned to color the graph.  
### How to use
```shell
python3 colorer.py prof.csv dep.csv part.csv optional_file_suffix 
```

Where part.csv contains a partition of a model:
- Layer name
- device ID

### Notes for input
The colorer supports up to eight (8) devices. For partitions that require more devices, please add more colors to line 51 in **colorer.py**:
```python
colors = ["red", "blue", "black", "yellow", "green", "purple", "white", "orange"]
```
---
# Plotters
There are several plotters located in child directories. See teh corresponding readme files in those directories for more info. 