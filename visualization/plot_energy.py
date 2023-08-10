# from matplotlib import rc

# rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

df1 = pd.read_csv('data/yolos-agx_2.csv')
df2 = pd.read_csv('data/yolos-agx_4.csv')
#df3 = pd.read_csv('data/yolos-agx_ultd.csv')


x1_list = []
for i in df1['bandwidth']:
    x1_list.append(i)

base = 95233.5336
baseBattery = 1
avg_battery = []

battery_plot_data1 = baseBattery * df1['device'] / (base + df1['energy']) / (baseBattery / base)
battery_plot_data2 = baseBattery * df2['device'] / (base + df2['energy']) / (baseBattery / base)
#battery_plot_data3 = baseBattery * df3['device'] / (base + df3['energy']) / (baseBattery / base)

avg_battery.append(battery_plot_data1[0])


fig, ax1 = plt.subplots()
fig.set_size_inches(5, 3)


w = (df1['bandwidth'][len(df1['bandwidth']) - 1] - df1['bandwidth'][0]) / len(df1) * 0.5
b0 = ax1.bar(x1_list, base, width=w, label='Energy: Computation', color=sns.xkcd_rgb["denim blue"])

b1 = ax1.bar(x1_list - w/4, df1['energy'], width=w/2, label='Energy: Communication', bottom=base, color=sns.xkcd_rgb["maize"])
b2 = ax1.bar(x1_list + w/4, df2['energy'], width=w/2, label='Energy: Communication', bottom=base, color=sns.xkcd_rgb["coral"])
#b3 = ax1.bar(x1_list + w/3, df3['energy'], width=w/3, label='Energy: Communication', bottom=base, color=sns.xkcd_rgb["green"])

ax2 = ax1.twinx()

line1, = ax2.plot(df1['bandwidth'], battery_plot_data1, color=sns.xkcd_rgb["pale red"], linestyle='-', label='Battery life')
line2, = ax2.plot(df2['bandwidth'], battery_plot_data2, color=sns.xkcd_rgb["navy"], linestyle='-', label='Battery life')
#line3, = ax2.plot(df3['bandwidth'], battery_plot_data3, color=sns.xkcd_rgb["green"], linestyle='-', label='Battery life')

p1 = ax2.scatter(df1['bandwidth'], battery_plot_data1, color=sns.xkcd_rgb["pale red"], marker='o', s=30, label='Battery life')
p2 = ax2.scatter(df2['bandwidth'], battery_plot_data2, color=sns.xkcd_rgb["navy"], marker='o', s=30, label='Battery life')
#p3 = ax2.scatter(df3['bandwidth'], battery_plot_data3, color=sns.xkcd_rgb["green"], marker='o', s=30, label='Battery life')

note = ax2.scatter([], [], marker='$1$', color="green", label="#device needed for optimization")

for i, j, d in zip(df1['bandwidth'], battery_plot_data1, df1["device"]):
        ax2.annotate('%s' % d, xy=(i, j), xytext=(-7, 3), textcoords='offset points', color="green")

for i, j, d in zip(df2['bandwidth'], battery_plot_data2, df2["device"]):
        ax2.annotate('%s' % d, xy=(i, j), xytext=(-7, 3), textcoords='offset points', color="green")

# for i, j, d in zip(df3['bandwidth'], battery_plot_data3, df3["device"]):
#         ax2.annotate('%s' % d, xy=(i, j), xytext=(-7, 3), textcoords='offset points', color="red")


if min(battery_plot_data1) > 0:
    ax2.set_ylim(ymin=0, ymax=(max(battery_plot_data1 + 2)))
    
ax2.set_ylabel("Battery life", fontsize=12)
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1fx'))
ax1.set_xlabel("Bandwidth (Mbps)", fontsize=12)
ax1.set_ylabel("Energy (mJ)", fontsize=12)


ax2.yaxis.label.set_color(line1.get_color())
ax1.yaxis.label.set_color('black')

ax2.tick_params(axis='y', colors=line1.get_color())
ax1.tick_params(axis='y', colors='black')

ax1.ticklabel_format(axis='y', style='sci', scilimits=(4, 4))



handles = [p1, p2, b0, b1, b2, note]
labels = ["Battery life (2GB)", "Battery life (4GB)", "Energy: Computation", "Energy: Communication (2GB)", "Energy: Communication (4GB)", "#device needed for optimization"]
plt.legend(handles=handles, labels = labels, loc="upper center", prop={"size" : 7}, bbox_to_anchor=(0.52, 1.15), ncol=2, fancybox=True)
#ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=3, fancybox=True)

plt.grid()
plt.savefig(f"energy.png", bbox_inches='tight', dpi=100)