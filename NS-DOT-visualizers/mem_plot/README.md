## Per layer memory consumption plotter
This is a plotter that generates a bar graph that shows the per layer memory consumption (in MB) of a model. 

To use the plotter, run
```shell
python3 plotter_memory.py
```

### Inputs
The input is a csv file that has two columns: model name, and  memory consumption in MB. 

Examples are given in the directory.

### Configuration
Configurations are at the beginning of the plotter. 

### Outputs
The output graphs are stored in `plots` directory.

---

## Per layer memory consumption plotter (combined)

This is a plotter that generates a bar graph that shows the total memory consumption (in MB) of several models.

To use the plotter, run
```shell
python3 plotter_memory_combined.py
```
Configurations are at the beginning of the plotter. 
