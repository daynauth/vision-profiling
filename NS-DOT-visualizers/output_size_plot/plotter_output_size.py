import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mem_plot.plotter_memory import config_bar_width

config = [
    "prof_faster",
    "prof_yolov4",
    "prof_yolox",
    "prof_yolor",
    "prof_yolor_merge",
]

config_graph_width = {
    'prof_faster': 7,
    'prof_yolov4': 11,
    'prof_yolox': 7,
    'prof_yolor': 20,
    'prof_yolor_merge': 11,
}

for filename in config:

    fig, ax1 = plt.subplots()
    fig.set_size_inches(config_graph_width[filename], 2.5)
    df_year = pd.read_csv("output_size_plot/" + filename + "_out.csv")
    x1_list = []
    for i in df_year['layer']:
        x1_list.append(i)

    b0 = ax1.bar(x1_list, df_year['size'], width=config_bar_width[filename], label='Output Data size',
                 color=sns.xkcd_rgb["dull green"])

    ax1.set_ylabel("Size (MB)", fontsize=13)
    # ax1.set_title(filename + ' per layer output size', fontsize=14)

    # Set colors for y-axis tags
    ax1.yaxis.label.set_color("black")

    # Set colors for y-axis marks
    ax1.tick_params(axis='y', colors="black")
    plt.xticks(rotation=90)

    # Set legends
    plt.legend(handles=[b0], loc='best', prop={'size': 13})
    plt.grid(axis='y', linestyle='--')
    for label in (ax1.get_yticklabels()):
        label.set_fontsize(14)
    plt.savefig(f"output_size_plot/plots/{filename}_out.png", bbox_inches='tight', dpi=100)
