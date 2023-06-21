import pandas as pd
import matplotlib.pyplot as plt

def cleanup_data(df):
    df = df.drop(columns=['cpu_mem', 'cuda_mem', 'size', 'macs'])
    df = df.drop([0, len(df)-1])
    return df


def plot_bars(df, intput1_name = 'rtx', input2_name = 'agx'):
    fig = plt.figure()
    ax = fig.add_subplot(111) # Create matplotlib axes
    ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

    width = 0.4
    df = df.set_index('layer_name')
    df.rtx_time.plot(kind='bar', color='red', ax=ax, width=width, position=1, x='layer_name')
    df.agx_time.plot(kind='bar', color='blue', ax=ax2, width=width, position=0, x='layer_name')



    ax.set_ylabel('RTX 3090 time (ms)')
    ax2.set_ylabel('Jetson AGX time (ms)')
    ax.set_xlabel('Layer')
    ax.set_title('AGX vs RTX')

    plt.tight_layout()
    plt.savefig('agx_vs_rtx.png')

input1 = 'prof_rtx.csv'
input2 = 'prof_cpu.csv'

input1_name = input1.split('.')[0].split('_')[1]
input2_name = input2.split('.')[0].split('_')[1]

df1 = pd.read_csv(input1)
df2 = pd.read_csv(input2)

df1 = cleanup_data(df1)
df2 = cleanup_data(df2)

df1_time = f'{input1_name}_time'
df2_time = f'{input2_name}_time'

df1 = df1.rename(columns={'time': df1_time})
df2 = df2.rename(columns={'time': df2_time})

df2 = df2.drop(columns=['layer_name'])
df = pd.concat([df1, df2], axis=1)

print(df)

df['speed_up'] = df[df2_time] / df[df1_time]
print(df)

#df = df.drop([1, 2])

#combine rtx and agx data

#plot speed up
df.plot(kind='bar', x='layer_name', y='speed_up')
plt.tight_layout()
plt.savefig(f'speed_up_{input1_name}_vs_{input2_name}.png')
df.to_csv(f'speed_up_{input1_name}_vs_{input2_name}.csv')

#plot_bars(df, input1_name, input2_name)

