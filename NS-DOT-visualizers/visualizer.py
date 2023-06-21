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


if len(sys.argv) < 3:
    print(">>> Incorrect usage, check README for instructions.")
    print(">>> Quitting.")
    exit(1)

# ==============================================
# Begin table data processing
# ==============================================

layer_table_path = sys.argv[1]
dependency_table_path = sys.argv[2]
if len(sys.argv) == 4:
    suffix = sys.argv[3]
else:
    suffix = ""

layer_table_list = table2list(layer_table_path)
dependency_table_list = table2list(dependency_table_path)

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
        print(f"{entry[0]}[label=\"{entry[0]}\\n{entry[1]}ms\\n{entry[3]}MB\"]")

    # layer dependency (edge)
    for entry in dependency_table_list:
        src = entry[0]
        dst = entry[1]
        i, e = index_2d(layer_table_list, src)
        src_data_size = layer_table_list[i][4]  # the size of layer output in mb
        print(f"{src} -- {dst}[label=\"{src_data_size}MB\"];")

    print("}")
    sys.stdout = original_stdout  # Reset the standard output to its original value

# ==============================================
# Begin DOT render
# ==============================================
