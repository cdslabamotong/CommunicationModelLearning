import copy
import os.path
import networkx as nx
import math
import csv

import torch
from torch.utils.data import Dataset, DataLoader
import time

# import practice
from ComplexPytorch.ComplexLayers import *
import torch.nn as nn
from ComplexPytorch.ComplexFunctions import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import zero_one_loss
from prettytable import PrettyTable


class ReadSample:

    def ReadGraph(ThresholdFileName, WeightsFileName):
        thresholds_file = open(ThresholdFileName, 'r')
        weights_file = open(WeightsFileName, 'r')
        g = nx.DiGraph()
        for count, line in enumerate(thresholds_file):
            if (count % 2) == 0:
                node = int(line.strip('\n'))
                g.add_node(node)
            if (count % 2) != 0:
                threshold_list = line.strip('\n').split(',')
                for count, ele in enumerate(threshold_list):
                    g.nodes[node][f'threshold_{count+1}'] = float(ele)
        for count, line in enumerate(weights_file):
            if (count % 2) == 0:
                edge = line.strip('\n')
                edge = eval(edge)
                g.add_edge(edge[0], edge[1])
            if (count % 2) != 0:
                weight_list = line.strip('\n').split(',')
                for count, ele in enumerate(weight_list):
                    g.edges[edge][f'weight_{count+1}'] = float(ele)
        return g

    def ReadWholeSample(NumSamples, SamplesFilePath):
        input_real = []
        output_real = []
        # dataset
        for sample in range(NumSamples):
            file_path = os.path.join(SamplesFilePath, f'sample{sample}.txt')
            sample_file = open(file_path, 'r').readlines()
            for count, line in enumerate(sample_file):
                if count == 1:
                    input_real.append(eval(line.strip('\n')))
                if count + 1 == len(sample_file):
                    output_real.append(eval(line.strip('\n')))
        a = len(output_real)
        b = len(output_real[0])

        input_real_part = torch.tensor(input_real).to(torch.float32)
        output_real_part = torch.tensor(output_real).to(torch.float32)
        ima_part = torch.zeros((a, b), dtype=torch.float32)
        input_np = torch.complex(input_real_part, ima_part)
        output_np = torch.complex(output_real_part, ima_part)

        data = torch.cat((input_np, output_np), 1)
        return data


class AdjacencyMatrix:
    def __init__(self, NumNodes, NumWorms, graph):
        self.NumWorms = NumWorms
        self.NumNodes = NumNodes
        self.inputsize = NumWorms * NumNodes
        self.outputsize = NumWorms * NumNodes
        self.graph = graph
        self.Num_StageTwo = math.log2(NumWorms)

    def StageOneTools(self, value):
        matrix = np.ndarray((self.NumWorms, self.NumWorms), dtype=int)

        if value == 1:
            matrix = np.identity(self.NumWorms, dtype=int)
            # matrix = matrix * 100
            # print(matrix)
        if value == 0:
            matrix = np.zeros((self.NumWorms, self.NumWorms), dtype=int)

        return matrix

    def StageOne(self):
        a = nx.adjacency_matrix(self.graph).todense()

        x, y = a.shape
        # print(x, y)
        final_list = []

        for i in range(x):

            matrix_x = self.StageOneTools(a[i, 0])

            for j in range(1, y):
                matrix_one_rest = self.StageOneTools(a[i, j])
                matrix_x = np.insert(matrix_x, [j * self.NumWorms], matrix_one_rest, axis=1)
            final_list.append(matrix_x)

        final_array = final_list[0]
        for ele_idx in range(1, x):
            final_array = np.insert(final_array, [ele_idx * self.NumWorms], final_list[ele_idx], axis=0)

        # print(final_array.shape)
        b = final_array.reshape((self.outputsize, self.outputsize))
        x, y = b.shape
        print(x, y)
        for i in range(x):
            b[i][i] = 1000.

        c = np.zeros((self.outputsize, self.outputsize))
        real_part = torch.from_numpy(b)
        real_part = real_part.to(torch.float32)
        ima_part = torch.from_numpy(c)
        ima_part = ima_part.to(torch.float32)
        weight_np = torch.complex(real_part, ima_part)
        weight_np.requires_grad = True
        # print(weight_np)
        return weight_np

    def StageTwo_comparison_layer1(self, comparison_step):
        # print(self.outputsize)
        # print(pow(2, comparison_step))
        input_size = int(self.outputsize/(pow(2, comparison_step)))
        # print(f'input_size:{input_size}')
        output_size = input_size

        weight_matrix = np.identity(input_size)
        for i in range(self.NumNodes):
            # print(i)
            weight_matrix[i*2+1][i*2] = -1

        real_part = torch.from_numpy(weight_matrix).to(torch.float32)
        ima_part = torch.zeros((input_size, output_size), dtype=torch.float32)
        weight_np = torch.complex(real_part, ima_part)
        return weight_np

    def StageTwo_comparison_layer2(self, comparison_step):
        input_size = int(self.outputsize/(pow(2, comparison_step)))
        output_size = int(input_size/2)
        weight_matrix = np.zeros((input_size, output_size))
        for i in range(self.NumNodes):
            weight_matrix[i*2][i] = 1
            weight_matrix[i*2+1][i] = 1
        real_part = torch.from_numpy(weight_matrix).to(torch.float32)
        ima_part = torch.zeros((input_size, output_size), dtype=torch.float32)
        weight_np = torch.complex(real_part, ima_part)
        weight_np.requires_grad = True

        return weight_np

    def RecoveryOne(self, shape_b):
        input_size = shape_b
        output_size = shape_b

        weight_matrix = np.identity(input_size)
        for i in range(input_size):
            weight_matrix[i][i] = -1

        ima_part = torch.from_numpy(weight_matrix).to(torch.float32)
        real_part = torch.zeros((input_size, output_size), dtype=torch.float32)

        weight_np = torch.complex(real_part, ima_part)
        weight_np.requires_grad = True

        return weight_np


    def RecoveryTwo(self):
        input_size = self.NumNodes
        output_size = self.outputsize

        weight_matrix = np.zeros((input_size, output_size))
        for i in range(self.NumNodes):
            for j in range(self.NumWorms):
                weight_matrix[i][i*self.NumWorms + j] = 1.

        real_part = torch.from_numpy(weight_matrix).to(torch.float32)
        ima_part = torch.zeros((input_size, output_size), dtype=torch.float32)
        weight_np = torch.complex(real_part, ima_part)
        weight_np.requires_grad = True

        return weight_np


    def RecoveryThree(self):
        input_size = self.inputsize
        output_size = self.outputsize

        weight_matrix = np.zeros((input_size, output_size))
        for i in range(self.NumNodes):
            for j in range(self.NumWorms):
                for k in range(self.NumWorms):
                    if j <= k:
                        weight_matrix[i*self.NumWorms + j][i*self.NumWorms + k] = 1
                    else:
                        if j > k:
                            weight_matrix[i*self.NumWorms + j][i*self.NumWorms + k] = -1

        real_part = torch.from_numpy(weight_matrix).to(torch.float32)
        ima_part = torch.zeros((input_size, output_size), dtype=torch.float32)
        weight_np = torch.complex(real_part, ima_part)
        weight_np.requires_grad = True

        return weight_np


    def StageOneWormsIndex(self):
        WormsIndex_matrix = np.zeros((self.inputsize, 1))
        for i in range(self.NumNodes):
            for j in range(self.NumWorms):
                WormsIndex_matrix[i*self.NumWorms + j] = j+1
                # print(j+1)
        ima_part = torch.from_numpy(WormsIndex_matrix).to(torch.float32)
        real_part = torch.zeros((self.inputsize, 1), dtype=torch.float32)
        StageOneWormsIndex_tensor = torch.complex(real_part, ima_part)
        return StageOneWormsIndex_tensor

    def Threshold_Tensor(self):
        threshold_matrix = np.ndarray((self.inputsize, 1))
        for i in range(self.NumNodes):
            for j in range(self.NumWorms):
                threshold_matrix[i*self.NumWorms + j] = -self.graph.nodes[i][f'threshold_{j+1}']

        threshold_tensor = torch.from_numpy(threshold_matrix).to(torch.float32)
        return threshold_tensor

    def RecoveryWormsIndex_Tensor(self):
        x = np.zeros((self.outputsize, 1))
        for i in range(self.NumNodes):
            for j in range(self.NumWorms):
                x[i * self.NumWorms + j][0] = j+1
        x = torch.from_numpy(x).to(torch.float32)
        return x


class WormsPropagationData(Dataset):
    def __init__(self, data, NumNodes, NumWorms):

        # data loading
        self.x_data = data[:, :(NumNodes * NumWorms)]
        self.y_data = data[:, (NumNodes * NumWorms):]
        self.data_size = int(self.x_data.shape[0])

    def __getitem__(self, index):
        # dataset[0]
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.data_size


class ComplexNN(nn.Module):
    def __init__(self, NumNode, NumWorms, graph, propagation_step):
        super(ComplexNN, self).__init__()
        self.NumNodes = NumNode
        self.NumWorms = NumWorms
        self.InputSize = NumNode * NumWorms
        self.OutputSize = self.InputSize
        self.PropagationStep = propagation_step
        self.adjacency = AdjacencyMatrix(NumNode, NumWorms, graph)
        self.NumComparison = self.adjacency.Num_StageTwo
        self.ThresholdTensor = self.adjacency.Threshold_Tensor()
        self.StageOneMatrix = self.adjacency.StageOne()

        self.StageOne_Linear = ComplexLinear(self.InputSize, self.OutputSize)
        nn.init.uniform_(self.StageOne_Linear.fc_r.weight, 0., 1.)
        nn.init.constant_(self.StageOne_Linear.fc_i.weight, 0.)
        self.StageOne_Linear.fc_r.weight = nn.Parameter(self.StageOne_Linear.fc_r.weight * self.StageOneMatrix.real)
        self.StageOne_Linear.fc_i.weight.requires_grad = False

        self.StageOneThreshold = ModuleStageOne()
        self.StageTwo = ModuleStageTwo()
        self.Recovery = ModuleRecover()
        self.CReLu = ComplexReLU()
        self.Sigmoid = ComplexSigmoid()
        self.tanh = ComplexTanh()

    def forward(self, x):
        for propagation in range(self.PropagationStep - 1):
            ### Stage One
            a, b = x.shape
            x = self.StageOne_Linear(x)
            m, n = self.adjacency.StageOneWormsIndex().shape
            adj = self.adjacency.StageOneWormsIndex().resize(m)
            adj = adj.repeat(a).resize(a, m)
            # print(adj)
            x = x + adj
            x = self.StageOneThreshold(x, self.ThresholdTensor)

            ## Stage Two
            for step in range(int(self.adjacency.Num_StageTwo)):
                x = complex_matmul(x, self.adjacency.StageTwo_comparison_layer1(step))
                x = self.StageTwo(x)
                x = complex_matmul(x, self.adjacency.StageTwo_comparison_layer2(step))

            ## Format Adjustment
            shape_a, shape_b = x.shape
            x = complex_matmul(x, self.adjacency.RecoveryOne(shape_b))
            x = self.CReLu(x)

            x = complex_matmul(x, self.adjacency.RecoveryTwo())

            # a__, b__ = x.shape
            # wormsindex_tensor_real = self.adjacency.RecoveryWormsIndex_Tensor().resize(b__).repeat(a__).resize(a__, b__).to(torch.float32)
            # # print(wormsindex_tensor_real)
            # wormsindex_tensor_imag = torch.zeros(a__, b__).to(torch.float32)
            # wormsindex_tensor = torch.complex(wormsindex_tensor_real, wormsindex_tensor_imag)

            # print(wormsindex_tensor)
            # x = x - wormsindex_tensor
            x = self.Recovery(x, self.adjacency.RecoveryWormsIndex_Tensor())
            # x = self.tanh(x)

            x = complex_matmul(x, self.adjacency.RecoveryThree())
            # x = x - wormsindex_tensor
            # x = self.tanh(x)
            x = self.Recovery(x, self.adjacency.RecoveryWormsIndex_Tensor())

        x = self.StageOne_Linear(x)
        x = self.StageOneThreshold(x, self.ThresholdTensor)
        x = complex_sigmoid(x)
        return x


class MainRun_WriteResults():
    def __init__(self, OutputFilePath,
                 TrainSizeSet,
                 TestSizeSet,
                 SampleSize,
                 InputFilePath,
                 BatchSize,
                 NumEpoch,
                 LearningRate,
                 TrainStep,
                 NumWorkers,
                 NumNode,
                 NumWorms
                 ):
        self.OutputFilePath = OutputFilePath
        self.InputFilePath = InputFilePath
        self.TrainSizeSet = TrainSizeSet
        self.TestSizeSet = TestSizeSet
        self.SampleSize = SampleSize
        self.BatchSize = BatchSize
        self.NumEpoch = NumEpoch
        self.LearningRate = LearningRate
        self.TrainStep = TrainStep
        self.NumWorkers = NumWorkers
        self.NumNodes = NumNode
        self.NumWorms = NumWorms

    def Loss(self, a, b):
        # a: predicted, b: true
        # loss during training
        m = nn.CrossEntropyLoss(reduction='mean')
        error = m(a, b)

        # other metrics
        x, y = a.shape
        a_1 = np.zeros(shape=(x, y))
        for i in range(x):
            for j in range(self.NumNodes):
                torch_array = a[i, self.NumWorms * (j) : self.NumWorms * (j+1)]
                max_index = torch.argmax(torch_array).item()
                a_1[i][int(max_index) + self.NumWorms * (j)] = 1.
        # print(f'predicted: {a_1}')
        # print(f'true: {b_1}')


        b_1 = b.cpu()
        b_1 = b_1.detach().numpy()
        # print(f'true: {b_1}')

        f1 = f1_score(b_1, a_1, average='weighted', zero_division=0)
        precision = precision_score(b_1, a_1, average='weighted', zero_division=0)
        recall = recall_score(b_1, a_1, average='weighted', zero_division=0)
        # accuracy = zero_one_loss(b_1, a_1)
        accuracy_2 = 0
        for i in range(x):
            accuracy_1 = balanced_accuracy_score(b_1[i], a_1[i])
            accuracy_2 += accuracy_1
        accuracy = accuracy_2/x
        prediction_error = [f1, precision, recall, accuracy]

        return error, prediction_error

    def ReadSamples(self):
        Data = ReadSample.ReadWholeSample(self.SampleSize, self.InputFilePath)
        return Data

    def ReadGraph(self):
        weight_file = os.path.join(self.InputFilePath, 'Weights.txt')
        threshold_file = os.path.join(self.InputFilePath, 'Threshold.txt')
        graph = ReadSample.ReadGraph(threshold_file, weight_file)
        return graph

    def DataLoader(self, train_size, test_size, Data):

        train_data, train_rest = torch.split(Data, [train_size, self.SampleSize - train_size])
        test_data, rest_ = torch.split(train_rest, [test_size, self.SampleSize - train_size - test_size])

        train_dataset = WormsPropagationData(train_data, self.NumNodes, self.NumWorms)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.BatchSize,
                                  shuffle=True,
                                  num_workers=self.NumWorkers)

        test_dataset = WormsPropagationData(test_data, self.NumNodes, self.NumWorms)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=self.BatchSize,
                                 shuffle=True,
                                 num_workers=self.NumWorkers)
        return train_loader, test_loader

    def count_parameters_table_show(self, model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params

    def Main(self):
        torch.multiprocessing.set_start_method('spawn', force=True)
        Data = self.ReadSamples()
        graph = self.ReadGraph()
        
        results_file = os.path.join(self.OutputFilePath, f'Batch{self.BatchSize}Lr{self.LearningRate}Epoch{self.NumEpoch}.csv')
        results_record = open(results_file, 'w')
        results_writer = csv.writer(results_record, delimiter=',')
        results_writer.writerow(
            ['TrainSize', 'TestSize', 'Run', 'TrainError', 'f1', 'precision', 'recall', 'accuracy',
             'TestError', 'f1', 'precision', 'recall', 'accuracy', 'TimeStart', 'TimeEnd', 'TimePeriod'])

        for Train_size in self.TrainSizeSet:
            for Test_size in self.TestSizeSet:
                for run in range(5):
                    print(f'Run: {run}')

                    n_train_iterations = math.ceil(Train_size / self.BatchSize)
                    n_test_iterations = math.ceil(Test_size / self.BatchSize)
                    print(f'TRAINSIZE: {Train_size}')
                    print(f'TESTSIZE: {Test_size}')

                    train_loader, test_loader = self.DataLoader(Train_size, Test_size, Data)
                    CNN_model = ComplexNN(self.NumNodes, self.NumWorms, graph, self.TrainStep)
                    optimizer = torch.optim.SGD(CNN_model.parameters(), lr=self.LearningRate, momentum=0.9)

                    def train_run(model):
                        model.train()
                        for i, (inputs, outputs) in enumerate(train_loader):
                            optimizer.zero_grad()
                            predict_output = model(inputs)
                            loss, predict_error__ = self.Loss(predict_output.abs(), outputs.abs())
                            loss.backward()
                            optimizer.step()
                            # self.count_parameters_table_show(model)

                            if i + 1 == n_train_iterations:
                                print(f'-----------------Train: Step {i + 1}/{n_train_iterations}, Loss {loss.item()}')
                            return loss.item(), predict_error__

                    def test_run(model):
                        model.eval()
                        for i, (inputs, outputs) in enumerate(test_loader):
                            predict_output = model(inputs)
                            loss, prediction_error = self.Loss(predict_output.abs(), outputs.abs())
                            if i + 1 == n_test_iterations:
                                print(f'-----------------Test: Step {i + 1}/{n_test_iterations}, Loss {loss.item()}')
                            return loss.item(), prediction_error

                    for epoch in range(self.NumEpoch):
                        print(f'Epoch: {epoch + 1}/{self.NumEpoch}')

                        t_s = time.time()
                        t_start = time.localtime()
                        time_start = time.strftime('%H%M%S', t_start)

                        train_error, predict_error = train_run(CNN_model)
                        test_error, predict_error_1 = test_run(CNN_model)

                        t_e = time.time()
                        t_end = time.localtime()
                        time_end = time.strftime('%H%M%S', t_end)
                        if epoch == self.NumEpoch - 1:
                            results_writer.writerow([Train_size,
                                                 Test_size,
                                                 run,
                                                 train_error,
                                                 predict_error[0],
                                                 predict_error[1],
                                                 predict_error[2],
                                                 predict_error[3],
                                                 test_error,
                                                 predict_error_1[0],
                                                 predict_error_1[1],
                                                 predict_error_1[2],
                                                 predict_error_1[3],
                                                 time_start,
                                                 time_end,
                                                 t_e - t_s
                                                 ])

            results_writer.writerow(
                ['TrainSize', 'TestSize', 'Run', 'TrainError', 'f1', 'precision', 'recall', 'accuracy',
                 'TestError', 'f1', 'precision', 'recall', 'accuracy', 'TimeStart', 'TimeEnd', 'TimePeriod'])

