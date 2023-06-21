import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mem_plot.plotter_memory import config_bar_width

config = [
    "prof_faster",
    "prof_yolov4",
    "prof_yolox",
    "prof_yolor",
]

power_config = [
    # "nano_0",
    "nano_1",
    # "agx_0",
    # "agx_1",
    # "agx_2",
]

config_graph_width = {
    'prof_faster': 5,
    'prof_yolov4': 11,
    'prof_yolox': 7,
    'prof_yolor': 15,
}

for filename in config:
    for power in power_config:
        fig, ax1 = plt.subplots()
        fig.set_size_inches(config_graph_width[filename], 2)
        df_year = pd.read_csv("per_layer_time_plot/" + filename + "_layer.csv")
        x1_list = []
        for i in df_year['layer']:
            x1_list.append(i)

        b0 = ax1.bar(x1_list, df_year[power], width=config_bar_width[filename], label='Layer latency',
                     color=sns.xkcd_rgb["denim blue"])

        ax1.set_ylabel("Time (s)", fontsize=13)
        # ax1.set_title(filename + ' per layer time', fontsize=14)

        # Set colors for y-axis tags
        ax1.yaxis.label.set_color("black")

        # Set colors for y-axis marks
        ax1.tick_params(axis='y', colors="black")
        plt.xticks(rotation=90)
        for label in (ax1.get_yticklabels()):
            label.set_fontsize(14)
        # Set legends
        plt.legend(handles=[b0], loc='best', prop={'size': 13})
        plt.grid(axis='y', linestyle='--')
        plt.savefig(f"per_layer_time_plot/plots/{power}/{filename}_layer.png", bbox_inches='tight', dpi=100)
