## End-to-end execution time plotter (per hardware)
This is a plotter that generates a bar graph that shows the end-to-end execution time (in second) of several models on one hardware platforms. 

To use the plotter, run
```shell
python3 plotter_end2end.py
```

### Inputs
The input is a csv file that has two columns: model name and execution time in seconds. 

Examples are given in the directory.
### Configuration
Configurations are at the beginning of the plotter: 
```python
config = ["prof_agx1.csv",
          "prof_nano1.csv",
          ]
```

### Outputs
The output graphs are stored in `plots` directory. 