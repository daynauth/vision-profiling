import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

config = [
    "prof_faster",
    "prof_yolov4",
    "prof_yolox",
    "prof_yolor",

]

config_graph_width = {
    'prof_faster': 5,
    'prof_yolov4': 11,
    'prof_yolox': 7,
    'prof_yolor': 15,
}

config_bar_width = {
    'prof_faster': 0.6,
    'prof_yolov4': 0.6,
    'prof_yolox': 0.6,
    'prof_yolor': 0.6,
    'prof_yolor_merge': 0.6,
}

for filename in config:

    fig, ax1 = plt.subplots()
    fig.set_size_inches(config_graph_width[filename], 2)
    df_year = pd.read_csv("mem_plot/" + filename + "_mem.csv")
    x1_list = []
    for i in df_year['layer']:
        x1_list.append(i)

    b0 = ax1.bar(x1_list, df_year['mem'], width=config_bar_width[filename], label='Memory Consumption',
                 color=sns.xkcd_rgb["dark rose"])

    ax1.set_ylabel("Memory (MB)", fontsize=13)
    # ax1.set_title(filename + ' per layer memory', fontsize=14)

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
    plt.savefig(f"mem_plot/plots/{filename}_mem.png", bbox_inches='tight', dpi=100)
