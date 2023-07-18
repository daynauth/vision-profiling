import pygraphviz as pgv

colors = ["#fe7070", "#6db1ff", "#54c45e", "yellow", "green", 
          "purple", "white", "orange", "darkseagreen2", "brown", "pink", "gray", "cyan", "gold", "darkolivegreen1", "darkorchid1",
            "darkorange1", "darkslategray1", "darkturquoise", "darkviolet", "deeppink1", "deepskyblue1", "dodgerblue1", "firebrick1",
            "forestgreen", "gold1", "greenyellow", "hotpink", "indianred1", "khaki1", "lightblue1", "lightcoral", "lightcyan1", "lightgoldenrod1",
          ]

class Graph(pgv.AGraph):
    def __init__(self, thing=None, filename=None, data=None, string=None, handle=None, name="", strict=True, directed=False, **attr):
        super().__init__(thing, filename, data, string, handle, name, strict, directed, **attr)

    def add_node(self, n, **attr):
        if 'style' not in attr:
            attr['style'] = 'filled'

        super().add_node(n, **attr)
        return self.get_node(n)
    
    def add_edge(self, u, v, **attr):
        try:
            u_node = self.get_node(u)
            v_node = self.get_node(v)
            super().add_edge(u_node, v_node, **attr)
        except KeyError:
            print(f"Node {u} or {v} not found")
            exit(1)
