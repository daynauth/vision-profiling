import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

config = ["prof_faster_end.csv",
          "prof_yolor_end.csv",
          "prof_yolov4_end.csv",
          "prof_yolox_end.csv",
          ]

for filename in config:

    fig, ax1 = plt.subplots()
    df_year = pd.read_csv("end2end_plot_model/" + filename)
    x1_list = []
    for i in df_year['device']:
        x1_list.append(i)

    b0 = ax1.bar(x1_list, df_year["time"], width=0.07, label='End to end latency',
                 color=sns.xkcd_rgb["denim blue"])

    ax1.set_xlabel("Device & Power setting", fontsize=12)
    ax1.set_ylabel("Time (s)", fontsize=12)
    ax1.set_title(filename, fontsize=14)

    # Set colors for y-axis tags
    ax1.yaxis.label.set_color("black")

    # Set colors for y-axis marks
    ax1.tick_params(axis='y', colors="black")
    plt.xticks(rotation=70)

    # Set legends
    plt.legend(handles=[b0], loc='best')
    plt.savefig(f"end2end_plot_model/plots/{filename}.png", bbox_inches='tight', dpi=100)
