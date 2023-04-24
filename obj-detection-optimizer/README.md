# Profile Units
Time - ms

Mem - MB

Size - MB

# Obj-Detection-Optimizer

## 1. Basic Optimizer Block

```python
from optimizer import Optimizer
Optimizer(
    dep_filename="dep.csv",
    prof_filenames=[
        "prof.csv",
        "prof.csv",
        "prof.csv",
        "prof.csv",
        "prof.csv",
    ],
    bandwidth=2000,
    parallel=True,
    ignore_latency=True,
    iterations=1,
    dir="testcase/explore"
)
```

## 2. Attribute Explanation
### 1. dep_filename
* This csv file contains the dependency relation between layers of a network. 
* It has two columns: source, destination. Every entry represents an edge in the network.
### 2. prof_filenames
* This csv file contains the profiling result of every layer on a particular device.
* One entry in this list represents one device. For example, in the basic optimizer block above, there are five devices available. 
* It has five columns: layer_name, time,cpu_mem, cuda_mem, size, MACs
### 3. bandwidth
* The bandwidth of communication network between drones. The unit is MBps. 
### 4. parallel
* data-computation parallel. 
### 5. ignore_latency
* Whether to ignore transfer latency. Mainly for testing. 

## 3. Use Optimizer Wrapper
Please use `opt_wrapper.py` when optimizing the network. Make a copy of the prof and dep files to the root directory. 

You may modify the following part as needed. 
```python
bandwidth = 400
ignore_latency = False
iteration = 5
prof_filenames = [
        "prof.csv",
        "prof.csv",
        # "prof.csv",
        # "prof.csv",
        # "prof.csv",
        # "prof.csv",
        # "prof.csv",
        # "prof.csv",
    ]
benchmark = 114.46748520400001  # agx
# ...
```
