import networkx as nx

def read_graph():
    NumNode = 55
    g = nx.DiGraph()
    for i in range(NumNode):
        g.add_node(i)

    f = open('/home/cds/Documents/worms_propagation_model/Real_graph_data/raw_intel_lab_data', 'r').readlines()
    for line in f:
        line = line.strip('\n').split(' ')
        if len(line) == 4:
            if float(line[3]) >= 0.5:
                g.add_edge(int(line[1]), int(line[2]))
    return g, NumNode

def powerlaw():
    NumNode = 768
    g = nx.DiGraph()
    for i in range(NumNode):
        g.add_node(i)
    f = open('/home/cds/Documents/worms_propagation_model/Real_graph_data/power768_diffusionModel', 'r').readlines()
    for line in f:
        line = line.strip('\n').split(' ')
        g.add_edge(int(line[0]), int(line[1]))
    return g, NumNode