import os

from graph import Graph, colors
from model_data import ModelData

class Graph_Render:
    def __init__(self, G, data, xpos=0, ypos=0, fine=True, split=True):
        self.G = G
        self.G.layout("neato")
        self.G.graph_attr['labelloc'] = 't'
        self.fine = fine
        self.split = split
        self.data = data
        self.xpos = xpos
        self.ypos = ypos
        self.colors = colors
        self.num_layers = 12


    def _update_node_label(self, data, label):
        if data is not None:
            time = round(float(data.time), 2)
            if time > 0:
                label += f'\\n{time}s'

            cuda_mem = round(float(data.cuda_mem), 2)
            if cuda_mem > 0:
                label += f'\\n{float(data.cuda_mem):.2f}MB'
        return label
    
    def _update_node_color(self, data, color):
        if data is not None:
            device = self.data.get_partition(data.name).device
            if device is not None:
                color = self.colors[int(device)]
        else:
            color = '#ffdda6'
        return color
    

    def _update_edge_label(self, data):
        label = round(float(data.size), 2)
        if label > 0:
            return f'<<B>{label} MB</B>>'
        
        return None

    def _render_node(self, name, offset=0, increment = 0, **kwargs):
    
        self.ypos += increment

        if kwargs.get('shape') is None:
            kwargs['shape'] = 'box'

        layer = self.data.get_layer(name)


        if kwargs.get('label') is None:
            kwargs['label'] = name

        kwargs['label'] = self._update_node_label(layer, kwargs.get('label'))

        if kwargs.get('fillcolor') is None:
            kwargs['fillcolor'] = self._update_node_color(layer, kwargs.get('fillcolor'))

        self.G.add_node(name, pos=f'{self.xpos + offset},{self.ypos}!', **kwargs)

    def _render_edge(self, u, v, **kwargs):
        layer = self.data.get_layer(u)
        if layer is not None and kwargs.get('label') is None:
            label = self._update_edge_label(layer)
            if label is not None:
                kwargs['label'] = label

        self.G.add_edge(u, v, **kwargs)

    def render_encoder(self, id = 0):
        self.ypos = 0

        layer = f'layer_{id}_'
        input = layer + 'input'
        pre_norm = layer + 'Layer_Norm_Before'
        query = layer + 'Query'
        key = layer + 'Key'
        value = layer + 'Value'
        x1 = layer + 'x1'
        matmul = layer + 'mul'
        div = layer + 'div'
        softmax = layer + 'softmax'
        dropout = layer + 'dropout'
        context = layer + 'context'
        self_output = layer + 'Self_Attention_Output'
        add1 = layer + 'Residual_Connection_1'
        x2 = layer + 'x2'
        post = layer + 'Layer_Norm_After'
        intermediate = layer + 'Intermediate_Forward'
        output = layer + 'Output'
        add2 = layer + 'Residual_Connection_2'


        increment = 1.5

        self._render_node(input, label='Input')
        self._render_node(x1, height = 0, shape='point', offset=2)
        self._render_node(pre_norm, label='Layer Normalization', increment=increment)
        self._render_node(query, label = 'Query', increment=increment, offset = -1.5)
        self._render_node(key, label = 'Key')
        self._render_node(value, label = 'Value', offset = 1.5)
        self._render_node(matmul, label = 'Matmul', increment=increment, offset=-0.75)
        self._render_node(div, label = 'div', increment=increment, offset=-0.75)
        self._render_node(softmax, label = 'softmax', increment=increment, offset=-0.75)
        self._render_node(dropout, label = 'dropout', increment=increment, offset=-0.75)
        self._render_node(context, label='matmul', increment=increment)
        self._render_node(self_output, label = 'YoloS Self Output', increment=True)
        self._render_node(add1, label = 'Add', increment=increment)
        self._render_node(x2, height = 0, shape='point', offset=-2)
        self._render_node(post, label = 'Layer Normalization', increment=increment)
        self._render_node(intermediate, label = 'YoloS Intermediate', increment=increment)
        self._render_node(output, label = 'YoloS Output', increment=increment)
        self._render_node(add2, label = 'Add', increment=increment)


        self._render_edge(input, pre_norm)
        self._render_edge(input, x1, dir='none')
        self._render_edge(pre_norm, query)
        self._render_edge(pre_norm, key)
        self._render_edge(pre_norm, value)
        self._render_edge(query, matmul)
        self._render_edge(key, matmul)
        self._render_edge(matmul, div)
        self._render_edge(div, softmax)
        self._render_edge(softmax, dropout)
        self._render_edge(dropout, context)
        self._render_edge(value, context)
        self._render_edge(context, self_output)
        self._render_edge(self_output, add1)
        self._render_edge(add1, post)
        self._render_edge(add1, x2, dir='none')
        self._render_edge(post, intermediate)
        self._render_edge(intermediate, output)
        self._render_edge(output, add2)
        self._render_edge(x1, add1)
        self._render_edge(x2, output)

        G.add_subgraph(
            [input, pre_norm, query, key, value, x1, matmul, 
             div, softmax, dropout, context, self_output, add1, 
             x2, post, intermediate, output, add2],
            name=f'cluster_{id}',
            color='none',
            labelloc='b',
        )

        self.xpos += 5


    def render_layers(self):
        self.ypos = 0
        input = 'Input'
        embedding = 'Embedding'
        norm = 'Layer_Norm'
        classifier = 'Class_Labels_Classifier'
        box_predictor = 'Box_Predictor'

        self._render_node(input, label='Input')
        self._render_node(embedding, label='Embedding', increment=1)
        

        self.num_layers = 12
        for i in range(self.num_layers):
            self._render_node(f'layer_{i}', label=f'Layer {i}', increment=1)

            if i < self.num_layers - 1:
                self._render_node(f'layer_{i}_add_mid_position_embedding', increment=1)
                self._render_edge(f'layer_{i}', f'layer_{i}_add_mid_position_embedding')

            if i > 0:
                self._render_edge(f'layer_{i-1}_add_mid_position_embedding', f'layer_{i}')
                
        self._render_node(norm, label='Layer Normalization', increment=1)
        self._render_node(classifier, label='Class Labels Classifier', increment=1, offset=-1.5)
        self._render_node(box_predictor, label='Box Predictor', offset=1.5)


        self._render_edge(input, embedding)
        self._render_edge(embedding, 'layer_0')
        self._render_edge(f'layer_{self.num_layers - 1}', norm)
        self._render_edge(norm, classifier)
        self._render_edge(norm, box_predictor)
        self.xpos += 5


    def render_interpolation(self):
        self.ypos = 0
        layers = self.num_layers - 1
        if self.split:
            for i in range(layers):
                self._render_node(f'layer_{i}_mid_position_embedding', increment=2)
                self._render_edge(f'layer_{i}_mid_position_embedding', f'layer_{i}_add_mid_position_embedding')
        else:
            self._render_node('Interpolation', height = 6, increment=13, shape = 'record')
            for i in range(layers):
                self._render_edge('Interpolation', f'layer_{i}_add_mid_position_embedding')

        self.xpos += 5




    def render_legend(self):
        self.ypos = 0
        partitions = set([int(d.device) for d in self.data.partitions])

        increment = 0.75
        for p in partitions:
            self._render_node(f'device_{p}', label = f'device {p}', shape='box', increment=increment, width = 1.5, fillcolor=self.colors[p])
            #self._render_node(f'device_{p}', shape='box', label=f'Device {p}', fillcolor=self.colors[p])


            


def convert_to_bool(s):
    return True if s == 'true' else False


fine = convert_to_bool(os.environ.get('fine'))
split = convert_to_bool(os.environ.get('split'))
size = os.environ.get('MEM_SIZE', 2)

fine = True
split = True

G = Graph(name = "diagram", directed=True, strict=True, rankdir='LR', splines='ortho')


data = ModelData(fine=fine, split=split, size=size)
gr = Graph_Render(G, data, fine=fine, split=split)

if fine:
    layers = 12
    for i in range(layers):
        gr.render_encoder(i)


gr.render_layers()
gr.render_interpolation()
gr.render_legend()
G.write('encoder.dot')