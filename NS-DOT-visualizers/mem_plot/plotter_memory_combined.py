import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

config = [
    "faster",
    "yolov4",
    "yolox",
    "yolor",

]

mem_sum = []
for filename in config:
    df_year = pd.read_csv("prof_" + filename + "_mem.csv")
    sum = 0
    for i in df_year['mem']:
        sum += i
    mem_sum.append(sum)
fig, ax1 = plt.subplots()
fig.set_size_inches(5, 4)
# ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 3))
b0 = ax1.bar(config, mem_sum, width=0.5, label='Memory Consumption',
             color=sns.xkcd_rgb["dark rose"])

# ax1.set_xlabel("Object detection model", fontsize=15)
ax1.set_ylabel("Memory (MB)", fontsize=25)
# ax1.set_title(filename + ' per layer memory', fontsize=14)

# Set colors for y-axis tags
ax1.yaxis.label.set_color("black")

# Set colors for y-axis marks
ax1.tick_params(axis='y', colors="black")
for label in (ax1.get_yticklabels()):
    label.set_fontsize(25)

for label in (ax1.get_xticklabels()):
    label.set_fontsize(20)

# Set legends
# plt.legend(handles=[b0], loc='lower right', prop={'size': 20})
plt.grid(axis='y', linestyle='--')
plt.savefig(f"plots/combined_mem.png", bbox_inches='tight', dpi=100)
