## Per layer output size plotter
This is a plotter that generates a bar graph that shows the per layer output size (in MB) of a model. 

To use the plotter, run
```shell
python3 plotter_output_size.py
```

### Inputs
The input is a csv file that has two columns: layer name, and output tensor size in MB. 

Examples are given in the directory.

### Configuration
Configurations are at the beginning of the plotter. 

### Outputs
The output graphs are stored in `plots` directory. 