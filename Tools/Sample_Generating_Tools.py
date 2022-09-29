import copy
import random

class WormsPropagationModel:
    def __init__(self, graph, seed, NumWorms):
        self.graph = graph
        self.NumNodes = len(graph.nodes())
        self.NumEdges = len(graph.edges())
        self.seed = seed
        self.NumWorms = NumWorms

    def generate_seed_set(self, GraphInitial):
        graph = GraphInitial
        node_list = graph.nodes()
        seed_list = random.sample(node_list, self.seed)
        for node in graph.nodes():
            if node in seed_list:
                graph.nodes[node]['status'] = random.randint(1, self.NumWorms)
            else:
                graph.nodes[node]['status'] = 0
        return graph

    def add_attr_on_graph(self, GraphInitial):
        for worm in range(1, self.NumWorms+1):
            for node in GraphInitial.nodes():
                GraphInitial.nodes[node][f'threshold_{worm}'] = round(random.uniform(0, 5), 2)
            for edge in GraphInitial.edges():
                print(GraphInitial.degree[edge[1]])
                degree_in = GraphInitial.degree[edge[1]]
                GraphInitial.edges[edge][f'weight_{worm}'] = 1/(degree_in + worm)

        return GraphInitial

    def graph_initial(self, threshold_file, weights_file):
        # graph = self.generate_seed_set()
        graph = self.graph
        step_zero_graph = self.add_attr_on_graph(graph)
        self.ParametersWriteToFile(step_zero_graph, threshold_file, weights_file)
        return step_zero_graph

    def local_diffusion(self, graph_zero):
        # graph_zero = self.step_zero()
        graph_after_one_step = copy.deepcopy(graph_zero)
        for node in graph_zero.nodes:
            predecessors = list(graph_zero.predecessors(node))
            if len(predecessors) == 0:
                continue
            weight_sum_ls = []
            if graph_zero.nodes[node]['status'] != 0:
                continue

            node_weight_sum = float('-inf')
            for worm_index in range(1, self.NumWorms+1):
                sum_weight_one_worm = 0
                for parent_node in predecessors:
                    if graph_zero.nodes[parent_node]['status'] == worm_index:
                        sum_weight_one_worm += graph_zero.edges[(parent_node, node)][f'weight_{worm_index}']

                if sum_weight_one_worm >= graph_zero.nodes[node][f'threshold_{worm_index}']:
                    if sum_weight_one_worm >= node_weight_sum:
                        node_weight_sum = sum_weight_one_worm
                        graph_after_one_step.nodes[node]['status'] = worm_index
        return graph_after_one_step

    def LocalToGlobal(self, GraphInitial, sample_file):
        # graph_zero = self.step_zero()
        graph_zero = self.generate_seed_set(GraphInitial)
        self.SampleWriteToFile(graph_zero, 0, sample_file)
        # self.ParametersWriteToFile(graph_zero, threshold_file, weights_file)
        # print(f'initial')
        # print(*graph_zero.nodes.data(), sep='\n')
        graph = copy.deepcopy(graph_zero)
        for step in range(1, self.NumNodes+1):
            nodes_status = copy.deepcopy(graph.nodes.data())
            graph = self.local_diffusion(graph)
            new_nodes_status = copy.deepcopy(graph.nodes.data())
            # print(f'step{step}')
            # print(*graph.nodes.data(), sep='\n')
            self.SampleWriteToFile(graph, step, sample_file)
            if nodes_status == new_nodes_status:
                diffusion_step = step

                break
            # print(step)
        return graph

    def ConvertToBool(self, value):
        num_worms = self.NumWorms
        lst = [0] * num_worms
        if value == 0:
            lst = lst
        else:
            lst[value-1] = 1
        return lst

    def SampleWriteToFile(self, graph, step, SampleFileName):
        samplefile = open(SampleFileName, 'a')
        samplefile.write(str(step)+'\n')

        step_status = []
        for node in graph.nodes():
            for ele in self.ConvertToBool(graph.nodes[node]['status']):
                step_status.append(ele)

        samplefile.write(",".join(str(item) for item in step_status)+'\n')


    def ParametersWriteToFile(self, graph, ThresholdsFileName, WeightsFileName):
        thresholdsfile = open(ThresholdsFileName, 'a')
        for node in graph.nodes():
            node_threshold = []
            thresholdsfile.write(f'{node}'+'\n')
            for worm in range(self.NumWorms):
                node_threshold.append(graph.nodes[node][f'threshold_{worm+1}'])

            thresholdsfile.write(",".join(str(item) for item in node_threshold)+'\n')

        weightsfile = open(WeightsFileName, 'a')
        for edge in graph.edges():
            edge_weights = []
            weightsfile.write(f'{edge}'+'\n')
            for worm in range(self.NumWorms):
                edge_weights.append(graph.edges[edge][f'weight_{worm + 1}'])

            weightsfile.write(",".join(str(item) for item in edge_weights)+'\n')




