import pandas as pd
import os
import sys
import argparse

from matplotlib import rc

rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

from matplotlib import pyplot as plt


def get_mem_data(folder):
    file_dir = os.path.join(folder, 'mem_plot')


    #get all files in folder that starts with prof_ and ends with _mem.csv
    prefix = 'prof_'
    suffix = '_mem.csv'

    files = [f for f in os.listdir(file_dir) if f.startswith(prefix) and f.endswith(suffix)]

    #extract model names from file names
    models = [file[len(prefix):-len(suffix)] for file in files]
    sums = [pd.read_csv(os.path.join(file_dir, file))['mem'].sum() for file in files]

    df = pd.DataFrame({'model': models, 'mem': sums})

    return df


def get_type_data(folder, type):
    file = os.path.join(folder, 'end2end_plot_platform', f'prof_{type}.csv')
    df = pd.read_csv(file)
    return df
    

def plot_combined(folder, type = 'mem', save_dir = './', ext = 'png'):

    if type == 'mem':
        df = get_mem_data(folder)
        y = 'mem'
    else:
        df = get_type_data(folder, type)
        y = 'time'

    fig, ax1 = plt.subplots()
    fig.set_size_inches(3, 4)

    df.plot.bar(x='model', y=y, ax=ax1, color="darkred", width=0.6, legend=False)


    plt.xticks(rotation=0, ha='center', fontsize=10)
    plt.xlabel("")
    plt.grid(axis='y', linestyle='--')


    save_file  = ''
    if type == 'mem':
        ylabel = "Memory (MB)"
        save_file = os.path.join(save_dir, f'combined_mem.{ext}')
    else:
        ylabel = "Time (ms)"
        save_file = os.path.join(save_dir, f'{type}.{ext}')

    ax1.set_ylabel(ylabel, fontsize=20)
    plt.savefig(save_file, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    plots = {'mem' : {
        'name' : 'mem',
        'folder' : 'mem_plot',
        'color' : 'darkred',
        'column' : 'mem',
        'ylabel' : 'Memory (MB)',
        'suffix' : 'mem'
    }}


    argparser = argparse.ArgumentParser()
    argparser.add_argument('-f', '--file-dir', type=str, default='../NS-DOT-visualizers/')
    argparser.add_argument('-e', '--extension', type=str, default='png')
    argparser.add_argument('-s', '--save-dir', type=str, default='./')

    args = argparser.parse_args()

    types = ['mem', 'agx1', 'nano1']

    for type in types:
        plot_combined(args.file_dir, type, args.save_dir, args.extension)