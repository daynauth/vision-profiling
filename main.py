import torch
from detect import YoloV4, YoloX, FasterRCNN, YoloR, YoloS
from profiler import Profiler
import matplotlib.pyplot as plt
import pandas as pd

from hook import *
from record import *

def profile(model: torch.nn.Module, model_name: str = 'model', device: str = 'cpu') -> pd.DataFrame:
    model = model.eval().to(device)  # load model
    image = torch.rand(1, 3, 640, 640).to(device)  # load image
    hook = TimeHook()
    profiler = Profiler(model, model_name, hook)
    record = profiler.run(image)
    return record.to_dataframe()

def plot_line(df, model_name, scale: float = 1.9):
    fig, ax = plt.subplots()
    df.plot.line(y='time', ax=ax, figsize=(12, 5), marker='o', markersize=5)
    plt.legend(loc='upper left')

    avg = df['time'].mean()

    for k, v in df.iterrows():
        if v['time'] > scale * avg:
            ax.text(k, v['time'], f"{v['layer_type']}", color='red', )

    plt.ylabel('time (ms)')
    plt.xlabel('layers')
    plt.savefig(f'images/layer_time_{model_name}.png')

def generate_line_plots(models, models_names, device):
    for model, name in zip(models, models_names):
        df = profile(model, name, device)
        plot_line(df, name)

def get_layer_name(x):
    return x.split('.')[-1]


def group_layers(model_name, df, selected_layers):
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

def generate_grouped_layers(model, model_name, selected_layers, device):
    df = profile(model, model_name, device)
    df = group_layers(model_name, df, selected_layers)

    return df


def generate_stacked_bar_plot(models, models_names, selected_layers, device):
    dfs = []
    for model, name in zip(models, models_names):
        df = generate_grouped_layers(model, name, selected_layers, device)
        dfs.append(df)

    dfs = pd.concat(dfs)
    dfs.reset_index(inplace=True, drop=True)
    dfs = dfs.pivot(index='model_name', columns='layer_type', values='time')
    dfs.plot.bar(stacked=True, figsize=(12, 5))
    plt.ylabel('time (ms)')
    plt.xlabel('models')
    plt.xticks(rotation=0, ha='center')
    plt.savefig('images/layer_time_stacked_bar.png')


def plot_bar(df, model_name):
    df.plot.bar(x='layer_type', y='time', figsize=(8, 8))
    plt.ylabel('time (ms)')
    plt.xlabel('layers')
    plt.xticks(rotation=75, ha='right')
    plt.tight_layout()
    plt.savefig(f'images/layer_time_bar_{model_name}.png')

def generate_bar_plot(models, models_names, device):
    for model, model_name in zip(models, models_names):
        df = profile(model, model_name, device)
        df = df.groupby('layer_type').sum().drop('layer_name', axis=1)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'layer_type'}, inplace=True)
        plot_bar(df, model_name)

def all_time_plots():
    models = [YoloV4(), YoloX(), FasterRCNN(), YoloR(), YoloS()] #not efficient, but it's just a test, PLEASE FIX
    models_names = ['yolov4', 'yolox', 'faster_rcnn', 'yolor', 'yolos']
    selected_layers = (torch.nn.Conv2d, torch.nn.Linear)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    #generate all the plots
    print("Generating line plots...")
    generate_line_plots(models, models_names, device)
    print("Generating stacked bar plot...")
    generate_stacked_bar_plot(models, models_names, selected_layers, device)
    print("Generating bar plots...")
    generate_bar_plot(models, models_names, device)

def test():
    model = YoloS()
    name = 'yolos'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    df = profile(model, name, device)
    plot_line(df, name)

    # model = FasterRCNN()
    # model_name = 'faster_rcnn'
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # model = model.eval().to(device)  # load model

    # image = torch.rand(1, 3, 640, 640).to(device)  # load image

    # hook = MemoryHook()

    # profiler = Profiler(model, model_name, hook)
    # records = profiler.run(image)
    
    # #write all record to file
    # with open('memory.txt', 'w') as f:
    #     for record in records:
    #         f.write(str(record) + '\n')

    # df = record.to_dataframe()
    # print(df.head())

def main(start_test: bool = False):
    if start_test:
        test()
    else:
        all_time_plots()

if __name__ == '__main__':
    main(start_test = False)