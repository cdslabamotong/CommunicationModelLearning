from Tools.Sample_Generating_Tools import *
import networkx as nx
import os
from Real_graph_data import intel_lab_data
# Possible Choices: 'medium_graph', 'small_graph', 'large_graph', 'intel_lab_data', 'ERgraph'.
Graph_Name = 'ERgraph'
NumNode = 200
NumSeed = 50
NumWorms = 4
SampleSize = 1000


Graph_ = nx.erdos_renyi_graph(NumNode, 0.2, directed=True)
# Graph_ = intel_lab_data.powerlaw()[0]

# t = time.localtime()
# time_record = time.strftime('%H%M%S', t)
Path = os.getcwd()
file_path = os.path.join(Path, f'Samples/{Graph_Name}/Samples{SampleSize}Node{NumNode}_Seed{NumSeed}_Worms{NumWorms}')


if not os.path.exists(file_path):
    os.mkdir(file_path)


weight_file = os.path.join(file_path, 'Weights.txt')
threshold_file = os.path.join(file_path, 'Threshold.txt')


model = WormsPropagationModel(Graph_, seed=NumSeed, NumWorms=NumWorms)
GraphInitial = model.graph_initial(threshold_file, weight_file)

for sample in range(SampleSize):
    print(f'Sample{sample}')
    sample_file = os.path.join(file_path, f'sample{sample}.txt')
    model.LocalToGlobal(GraphInitial, sample_file)

