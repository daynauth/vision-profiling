import pandas as pd
import os
import sys
import argparse

from matplotlib import rc

#rc('font',**{'family':'serif','serif':['Palatino']})
rc('font',**{'family':'serif'})
#rc('text', usetex=True)

from matplotlib import pyplot as plt


model_width = {
    'yolor': 12,
    'yolox': 4,
    'yolov4': 12,
    'faster': 4,
}


def plot_graph(model, file_dir, save_dir, extension='png', plot_info=None):
    if plot_info is None:
        print("No plot info provided")
        return
    

    file = os.path.join(file_dir, plot_info['folder'], f"prof_{model}_{plot_info['suffix']}.csv")
    df = pd.read_csv(file)

    fig, ax1 = plt.subplots()
    fig.set_size_inches(model_width[model], 2)

    #use ax1 to plot bar chart with index as x values and column 'mem' as y values
    df.plot.bar(x='layer', y=plot_info['column'], ax=ax1, color=plot_info['color'], width=0.6, legend=False)

    if plot_info["name"] == "out":
        plt.ylabel(plot_info['ylabel'], fontsize=13)
    else:
        plt.ylabel(plot_info['ylabel'], fontsize=16)
    plt.xlabel("Layers", fontsize=16)
    plt.yticks(fontsize=16)


    plt.xticks([])
    plt.grid(axis='y', linestyle='--')
    file = os.path.join(save_dir, f"{model}_{plot_info['name']}_v2.{extension}")
    plt.savefig(file, bbox_inches='tight')




if __name__ == '__main__':
    models = ['yolor', 'yolox', 'yolov4', 'faster']
    plots = {'mem' : {
        'name' : 'mem',
        'folder' : 'mem_plot',
        'color' : 'darkgoldenrod',
        'column' : 'mem',
        'ylabel' : 'Memory (MB)',
        'suffix' : 'mem'
    }, 
    'time' : {
        'name' : 'time',
        'folder' : 'per_layer_time_plot',
        'color' : 'steelblue',
        'column' : 'nano_1',
        'ylabel' : 'Latency (s)',
        'suffix' : 'layer'
    },
    'out' : {
        'name' : 'out',
        'folder' : 'output_size_plot',
        'color' : 'olivedrab',
        'column' : 'size',
        'ylabel' : 'Output Size (MB)',
        'suffix' : 'out'
    }}


    argparser = argparse.ArgumentParser()
    argparser.add_argument('-f', '--file-dir', type=str, default='../NS-DOT-visualizers/')
    argparser.add_argument('-s', '--save-dir', type=str, default='./')
    argparser.add_argument('-e', '--extension', type=str, default='pdf')
    argparser.add_argument('-p', '--plot', type=str, default='mem')

    


    args = argparser.parse_args()


    for model in models:
        plot_graph(model, args.file_dir, args.save_dir, args.extension, plots[args.plot])

