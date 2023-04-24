from optimizer import Optimizer
from simulatorv2 import Simulator

bandwidth = 375
ignore_latency = False
iteration = 5
prof_filenames = [
        "prof.csv",
        "prof.csv",
        "prof.csv",
        "prof.csv",
        "prof.csv",
        "prof.csv",
        # "prof.csv",
        # "prof.csv",
    ]
# benchmark = 114.46748520400001  # agx
# benchmark = 194.446187  # agx no warm up
# benchmark = 274.2038  # agx time.time()
# benchmark = 47.999  # clarity32_new
# benchmark = 69.7  # clarity32
# benchmark = 139.144019769287  # nx
# benchmark = 331.0782  # cpu_vit
# benchmark = 1111.2275  # yolox_cpu on mac
# benchmark = 31.5581  # yolox clarity32
benchmark = 193.1997  # yolox agx
# benchmark = 5626.559550000001  # yolox nano


results = []
best = []
best_iter = []
r0 = []
r1 = []

opt0 = Optimizer(
    dep_filename="dep.csv",
    prof_filenames=prof_filenames,
    bandwidth=bandwidth,  # MBps
    ignore_latency=ignore_latency,
    iterations=iteration,
    benchmark=benchmark,
    reverse0=True,
    reverse1=True,
)
results.append(opt0.report())

opt1 = Optimizer(
    dep_filename="dep.csv",
    prof_filenames=prof_filenames,
    bandwidth=bandwidth,  # MBps
    ignore_latency=ignore_latency,
    iterations=iteration,
    benchmark=benchmark,
    reverse0=False,
    reverse1=True,
)
results.append(opt1.report())

opt2 = Optimizer(
    dep_filename="dep.csv",
    prof_filenames=prof_filenames,
    bandwidth=bandwidth,  # MBps
    ignore_latency=ignore_latency,
    iterations=iteration,
    benchmark=benchmark,
    reverse0=False,
    reverse1=False,
)
results.append(opt2.report())

opt3 = Optimizer(
    dep_filename="dep.csv",
    prof_filenames=prof_filenames,
    bandwidth=bandwidth,  # MBps
    ignore_latency=ignore_latency,
    iterations=iteration,
    benchmark=benchmark,
    reverse0=True,
    reverse1=False,
)
results.append(opt3.report())

for result in results:
    best.append(result[0])
    best_iter.append(result[1])
    r0.append(result[2])
    r1.append(result[3])

opt4 = Optimizer(
    dep_filename="dep.csv",
    prof_filenames=prof_filenames,
    bandwidth=bandwidth,  # MBps
    ignore_latency=ignore_latency,
    iterations=best_iter[best.index(min(best))],
    benchmark=benchmark,
    reverse0=r0[best.index(min(best))],
    reverse1=r1[best.index(min(best))],
)

print(f"\n\033[30;42m=========Result=========\033[0m")
print(f"Best result: {min(best)}")
print(f"Performance: {(benchmark - min(best)) / benchmark}")
print(f"Iteration: {best_iter[best.index(min(best))]}")
print(f"Setting: {r0[best.index(min(best))]}, {r1[best.index(min(best))]}")
