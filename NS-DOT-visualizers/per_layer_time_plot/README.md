## Per layer execution time plotter
This is a plotter that generates a bar graph that shows the per layer execution time (in second) of a model on one hardware and under one power mode. 

To use the plotter, run
```shell
python3 plotter_per_layer_time.py
```

### Inputs
The input is a csv file that has several columns: layer name, and several columns of execution time in seconds, under different settings. 

Examples are given in the directory.

### Configuration
Configurations are at the beginning of the plotter:

### Outputs
The output graphs are stored in `plots` directory. 