import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

config = ["prof_agx1.csv",
          "prof_nano1.csv",
          ]

for filename in config:

    fig, ax1 = plt.subplots()
    fig.set_size_inches(5, 4)
    df = pd.read_csv(filename)
    x1_list = []
    for i in df['model']:
        x1_list.append(i)

    b0 = ax1.bar(x1_list, df["time"], width=0.5, label='End to end latency',
                 color=sns.xkcd_rgb["denim blue"])

    # ax1.set_xlabel("Object detection model", fontsize=15)
    ax1.set_ylabel("Time (s)", fontsize=20)
    # ax1.set_title(filename + ' power mode 1', fontsize=20)

    # Set colors for y-axis tags
    ax1.yaxis.label.set_color("black")

    # Set colors for y-axis marks
    ax1.tick_params(axis='y', colors="black")
    plt.xticks(rotation=0)

    # Set legends
    # plt.legend(handles=[b0], loc='best', prop={'size': 20})
    for label in (ax1.get_yticklabels()):
        label.set_fontsize(25)

    for label in (ax1.get_xticklabels()):
        label.set_fontsize(20)

    plt.savefig(f"plots/{filename}1.png", bbox_inches='tight', dpi=100)
