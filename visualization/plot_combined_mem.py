import pandas as pd
import os
import sys
import argparse

from matplotlib import rc

#rc('font',**{'family':'serif','serif':['Palatino']})
rc('font',**{'family':'serif'})
# rc('text', usetex=True)

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
    

def plot_device(ax):
    y = [2000,2000,3000,4000]
    x = [1,2,3,4]

    ax.bar(x, y, width=0.6, color='darkslateblue', label = "devices")

    ax.set_title("Devices", verticalalignment='bottom', y=0, pad=-40)
    #ax.set_yticklabels([])




def plot_models(ax, df):
    y = 'mem'
    ylabel = "Model Size (GB)"

    plot_order = ["yolor", "faster", "yolov4", "yolox"]
    df = df.set_index("model").reindex(plot_order).reset_index()
    mapping = {
        "faster": "FRCNN",
        "yolor": "YOLOR",
        "yolov4": "YOLOv4",
        "yolox": "YOLOX"
    }
    df["model"] = df["model"].replace(mapping)
    df[y] /= 1000
 
    ax.bar(df['model'], df[y], color="darkgoldenrod", width=0.6, edgecolor="grey", linewidth=1.5)
    # ax.set_title("Models", verticalalignment='bottom', y=0, pad=-40)
    ax.set_ylabel(ylabel, fontsize=24)


def plot_combined(folder, type = 'mem', save_dir = './', ext = 'png'):
    df = get_mem_data(folder)


    fig, ax1 = plt.subplots()
    fig.set_size_inches(8, 4.5)
    
    plt.xlabel("")
    plt.grid(axis='y', linestyle='--')

    save_file  = ''


    #for ax in ax1:
    #    ax1.set_ylim([0, 7000])


    plot_models(ax1, df)
 #   plot_device(ax1[1])



    save_file = os.path.join(save_dir, f'combined_mem.{ext}')


    plt.subplots_adjust(wspace=0, hspace=0)

    #for ax in ax1:
    # ax1.grid(which='major', linestyle='--', linewidth='0.5', color='grey')

    ax1.tick_params(axis='both', which='major', labelsize=24)
    plt.xticks(rotation=0, ha='center', fontsize=24)

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
    argparser.add_argument('-e', '--extension', type=str, default='pdf')
    argparser.add_argument('-s', '--save-dir', type=str, default='./')

    args = argparser.parse_args()





    plot_combined(args.file_dir, 'mem', args.save_dir, args.extension)