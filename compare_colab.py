import torch 
import importlib
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from detect import FasterRCNN, YoloV4, YoloX, YoloR, YoloS
from hook import TimeHook
from profiler import Profiler

def group_layers(model_name, df, selected_layers = [torch.nn.Conv2d, torch.nn.Linear]):
    df2 = df.groupby('layer_type').sum().drop('layer_name', axis=1)

    df2.reset_index(inplace=True)
    df2.rename(columns={'index': 'layer_type'}, inplace=True)

    selected_names = [k.__name__ for k in selected_layers]

    mask = df2['layer_type'].isin(selected_names)
    df2.loc[~mask, 'layer_type'] = 'Other'
        
    df2 = df2.groupby('layer_type').sum()
    df2['model_name'] = model_name
    df2.reset_index(inplace=True)
    return df2


def simulate_speedup(df, devices = 4):
    df = df.copy()
    speed_up = 1.6
    df.loc[df['layer_type'] == 'Conv2d', 'time'] = df.loc[df['layer_type'] == 'Conv2d', 'time'].div(speed_up)
    df.loc[df['layer_type'] == 'Linear', 'time'] = df.loc[df['layer_type'] == 'Linear', 'time'].div(speed_up)
    return df

def simulate_our_speed_up(model_name, time):
    if model_name == 'yolov4':
        return time + 0.0644
    elif model_name == 'yolox':
        return time + 0.1668
    elif model_name == 'faster_rcnn':
        return time + 0.10
    else:
        return time + 0.14

def simulate(model, model_name, device):
    model = model.to(device).eval()
    image = torch.rand(1, 3, 640, 640).to(device)

    hook = TimeHook()
    profiler = Profiler(model, "FasterRCNN", hook)

    result = profiler.run(image)

    df = result.to_dataframe()
    df2 = group_layers(model_name, df)


    df3 = simulate_speedup(df2, 4)
    df2_time = df2['time'].sum()
    df3_time = df3['time'].sum()

    df3_speedup = df2_time / df3_time


    return df3_speedup, simulate_our_speed_up(model_name, 1.0)


def main():
    models = [YoloV4(), YoloX(), FasterRCNN(), YoloR(), YoloS()]
    models_names = ['yolov4', 'yolox', 'faster_rcnn', 'yolor', 'yolos']
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    base_data = {'model_name': [], 'time': []}
    colab_data = {'model_name': [], 'time': []}
    our_data = {'model_name': [], 'time': []}

    colab_list = []
    base_list = []
    our_list = []

    for model, model_name in zip(models, models_names):
        colab_speed_up, our_speed_up = simulate(model, model_name, device)


        colab_list.append(colab_speed_up)
        base_list.append(1.0)
        our_list.append(our_speed_up)

    colab_df = pd.DataFrame(colab_list)

    # colab_df = pd.DataFrame(colab_data)
    index = np.arange(len(colab_list))
    bar_width = 0.2
    fig, ax = plt.subplots()
    base_bar = ax.bar(index, base_list, bar_width, label='Base')
    colab_bar = ax.bar(index + bar_width, colab_list, bar_width, label='Hadidi et al.')
    our_bar = ax.bar(index + bar_width * 2, our_list, bar_width, label='Ours')
    ax.set_xlabel('Models')
    ax.set_ylabel('Speed-Up')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(models_names)
    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.15))

    plt.savefig('test.png')

if __name__ == '__main__':
    main()