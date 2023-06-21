import pandas as pd
import sys
import numpy as np


def table2list(table_path):
    df = pd.read_csv(table_path)
    for column_name in df.columns.values:
        df[column_name] = df[column_name].apply(lambda x: str(x).replace(u'\xa0', u''))
        df[column_name] = df[column_name].apply(lambda x: str(x).replace(u' ', u''))
    df_list = df.values.tolist()
    return df_list


def index_2d(data, search):
    for i, e in enumerate(data):
        try:
            return i, e.index(search)
        except ValueError:
            pass
    raise ValueError("{!r} is not in list".format(search))


def compute_num_param(size):
    size = size[1:-1]
    l = [int(x) for x in size.split(",")]
    return np.prod(l)


if len(sys.argv) < 4:
    print(">>> Incorrect usage, check README for instructions.")
    print(">>> Quitting.")
    exit(1)

# ==============================================
# Begin table data processing
# ==============================================

layer_table_path = sys.argv[1]
dependency_table_path = sys.argv[2]
partition_table_path = sys.argv[3]
if len(sys.argv) == 5:
    suffix = sys.argv[4]
else:
    suffix = ""

layer_table_list = table2list(layer_table_path)
dependency_table_list = table2list(dependency_table_path)
layer_partition_list = table2list(partition_table_path)
colors = ["red", "blue", "teal", "yellow", "green", 
          "purple", "white", "orange", "darkseagreen2", "brown", "pink", "gray", "cyan", "gold", "darkolivegreen1", "darkorchid1",
            "darkorange1", "darkslategray1", "darkturquoise", "darkviolet", "deeppink1", "deepskyblue1", "dodgerblue1", "firebrick1",
            "forestgreen", "gold1", "greenyellow", "hotpink", "indianred1", "khaki1", "lightblue1", "lightcoral", "lightcyan1", "lightgoldenrod1",
          ]

# ==============================================
# Begin DOT code generation
# ==============================================

original_stdout = sys.stdout
if suffix != "":
    dot_filename = f'result_DOT_code_{suffix}.dot'
else:
    dot_filename = 'result_DOT_code.dot'

with open(dot_filename, 'w') as f:
    sys.stdout = f  # Change the standard output to the file we created.
    print("graph {")

    # layout settings
    print("rankdir=LR")

    # layer info (node)
    for entry in layer_table_list:
        device = 0
        for entry_part in layer_partition_list:
            if entry_part[0] == entry[0]:
                device = entry_part[1]
        print(
            f"{entry[0]}[label=\"{entry[0]}\\n{entry[1]}ms\\n{entry[3]}MB\", style=filled, fillcolor=\"{colors[int(device)]}\"]")

    # layer dependency (edge)
    for entry in dependency_table_list:
        src = entry[0]
        dst = entry[1]
        i, e = index_2d(layer_table_list, src)
        src_data_size = layer_table_list[i][4]  # the size of layer output in mb
        # src_param_size = compute_num_param(src_data_size)
        # print(f"{src} -- {dst}[label=\"{src_data_size}\\n({src_data_param})\"];")
        print(f"{src} -- {dst}[label=\"{src_data_size}MB\"];")

    print("}")
    sys.stdout = original_stdout  # Reset the standard output to its original value

# ==============================================
# Begin DOT render
# ==============================================
