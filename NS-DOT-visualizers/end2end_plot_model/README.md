## End-to-end execution time plotter (per model)
This is a plotter that generates a bar graph that shows the end-to-end execution time (in second) of a model on different hardware platforms. 

To use the plotter, run
```shell
python3 plotter_end2end.py
```

### Inputs
The input is a csv file that has two columns: device type and execution time in seconds. 

Examples are given in the directory.
### Configuration
Configurations are at the beginning of the plotter: 
```python
config = ["prof_faster_end.csv",
          "prof_yolor_end.csv",
          "prof_yolov4_end.csv",
          "prof_yolox_end.csv",
          ]
```

### Outputs
The output graphs are stored in `plots` directory. 