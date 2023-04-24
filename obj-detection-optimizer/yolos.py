from optimizer import Optimizer

Optimizer(
    dep_filename="../dep.csv",
    prof_filenames=[
        "../prof.csv",
        "../prof.csv",
        "../prof.csv",
        "../prof.csv",
        "../prof.csv",
    ],
    bandwidth=2000,
    parallel=True,
    ignore_latency=True,
    iterations=1,
    dir="testcases/yolos"
)