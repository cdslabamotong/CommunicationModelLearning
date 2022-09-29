from multiprocessing.spawn import old_main_modules
import os.path
import networkx as nx
import math
import csv
import numpy as np
from prettytable import PrettyTable
from torch.utils.data import Dataset, DataLoader
import time
import torch.nn as nn
# from ComplexPytorch.ComplexFunctions import *
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import balanced_accuracy_score
import prettytable
import random
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class ReadSample:

    def ReadGraph(self, ThresholdFileName, WeightsFileName):
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


        input_real_part = torch.tensor(input_real).to(torch.float32)
        output_real_part = torch.tensor(output_real).to(torch.float32)

        data = torch.cat((input_real_part, output_real_part), 1)
        return data


class SamplePreparation_SizeN():
    def __init__(self, SampleFilePath, NumSamples, NumNode, NumWorms):
        self.SampleFilePath = SampleFilePath
        self.NumSamples = NumSamples
        self.Numnode = NumNode
        self.Numworm = NumWorms
    
    def ConvertBinaryToIndex(self, StepList, NumNode, NumWorm):
        # len(StepList) = NumNode * NumWorm
        NewStep = []
        for i in range(NumNode):
            one_node_list = []
            for j in range(NumWorm):
                one_node_list.append(StepList[NumWorm * i + j])
            if 1 in one_node_list:
                # print(one_node_list.index(1) + 1)
                NewStep.append(one_node_list.index(1) + 1)
            if 1 not in one_node_list:
                NewStep.append(0)
        # print(NewStep)
        return NewStep

    def Baseline_ReadWhoSample(self):
        input = []
        output = []

        for sample in range(self.NumSamples):
            file_path = os.path.join(self.SampleFilePath, f'sample{sample}.txt')
            sample_file = open(file_path, 'r').readlines()
            for count, line in enumerate(sample_file):
                if count == 1:
                    old_step_list = eval(line.strip('\n'))
                    new_step_list = self.ConvertBinaryToIndex(old_step_list, self.Numnode, self.Numworm)
                    input.append(new_step_list)

                if count + 1 == len(sample_file):
                    old_step_list = eval(line.strip('\n'))
                    new_step_list = self.ConvertBinaryToIndex(old_step_list, self.Numnode, self.Numworm)
                    output.append(new_step_list)

        input_ = torch.tensor(input).to(torch.float32)
        output_ = torch.tensor(output).to(torch.float32)
        data = torch.cat((input_, output_), 1)
        return data

    def Baseline_ReadWhoSample_Numpy(self):
        input = []
        output = []

        for sample in range(self.NumSamples):
            file_path = os.path.join(self.SampleFilePath, f'sample{sample}.txt')
            sample_file = open(file_path, 'r').readlines()
            for count, line in enumerate(sample_file):
                if count == 1:
                    old_step_list = eval(line.strip('\n'))
                    new_step_list = self.ConvertBinaryToIndex(old_step_list, self.Numnode, self.Numworm)
                    input.append(new_step_list)
                    # print(len(new_step_list))

                if count + 1 == len(sample_file):
                    old_step_list = eval(line.strip('\n'))
                    new_step_list = self.ConvertBinaryToIndex(old_step_list, self.Numnode, self.Numworm)
                    output.append(new_step_list)
        # print(len(input))
        input_ = np.asarray(input)
        output_ = np.asarray(output)
        return input_, output_

    def ReadGraph(self, ThresholdFileName, WeightsFileName):
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

    def SVM_Data_Preparation(self, graph, inputs, outputs):
        '''

        :param graph: networkx graph
        :param inputs: numpy.array, shape(sample_size, num_nodes)
        :param outputs: numpy.array, shape(sample_size, num_nodes)
        :return: inputs_array, output_array
        '''

        SampleSize, NumNode = inputs.shape
        node_inneighbor_dict = {}
        for node in graph.nodes:
            predecessors = list(graph.predecessors(node))
            node_inneighbor_dict[node] = predecessors

        input_dict = {}
        output_dict = {}
        # print(outputs.shape)
        for node in range(1, NumNode):
            # print(node)
            indices = node_inneighbor_dict[node]
            node_input = np.take(inputs, indices, axis=1)
            input_dict[node] = node_input

            output_dict[node] = np.take(outputs, node, axis=1)

        return input_dict, output_dict


class RegularNN(nn.Module):
    def __init__(self, NumNode, NumWorms, propagation_step):
        super(RegularNN, self).__init__()
        self.InputSize = NumNode * NumWorms
        self.OutputSize = self.InputSize
        self.PropagationStep = propagation_step

        self.Linear1 = nn.Linear(self.InputSize, self.OutputSize, bias=False)
        self.Linear2 = nn.Linear(self.InputSize, self.OutputSize, bias=False)
        self.Linear3 = nn.Linear(self.InputSize, self.OutputSize, bias=False)
        self.Linear4 = nn.Linear(self.InputSize, self.OutputSize, bias=False)
        self.ReLu = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()

        # nn.init.uniform_(self.Linear1.weight, 0., 1.)
        # nn.init.uniform_(self.Linear2.weight, 0., 1.)
        # nn.init.uniform_(self.Linear3.weight, 0., 1.)
        # nn.init.uniform_(self.Linear4.weight, 0., 1.)

    def forward(self, x):
        for step in range(self.PropagationStep):
            x = self.Linear1(x)
            x = nn.functional.normalize(x, dim=1)
        return x


class WormsPropagationData_SizeN(Dataset):
    def __init__(self, data, NumNodes):

        # data loading
        self.x_data = data[:, :(NumNodes)]
        self.y_data = data[:, (NumNodes):]
        self.data_size = int(self.x_data.shape[0])

    def __getitem__(self, index):
        # dataset[0]
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.data_size


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


class RegularNN_WriteResults():
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

    def ReadSamples(self):
        Data = ReadSample.ReadWholeSample(self.SampleSize, self.InputFilePath)
        return Data

    def Loss(self, predicted, true):

        # print(f'predicted{predicted}')
        # print(f'true{true}')
        l = torch.nn.MSELoss(reduction='mean')
        loss = l(predicted, true)

        x, y = predicted.shape
        a_1 = np.zeros(shape=(x, y))
        for i in range(x):
            for j in range(self.NumNodes):
                torch_array = predicted[i, self.NumWorms * (j) : self.NumWorms * (j+1)]
                max_index = torch.argmax(torch_array).item()
                a_1[i][int(max_index) + self.NumWorms * (j)] = 1.
        b_1 = true.cpu()
        b_1 = b_1.detach().numpy()

        f1 = f1_score(b_1, a_1, average='weighted', zero_division=0)
        precision = precision_score(b_1, a_1, average='weighted', zero_division=0)
        recall = recall_score(b_1, a_1, average='weighted', zero_division=0)
        accuracy_2 = 0
        for i in range(x):
            accuracy_1 = balanced_accuracy_score(b_1[i], a_1[i])
            accuracy_2 += accuracy_1
        accuracy = accuracy_2/x


        prediction_error = [f1, precision, recall, accuracy]

        return loss, prediction_error

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
        results_file = os.path.join(self.OutputFilePath, f'Batch{self.BatchSize}Lr{self.LearningRate}Epoch{self.NumEpoch}.csv')
        results_record = open(results_file, 'w')
        results_writer = csv.writer(results_record, delimiter=',')
        results_writer.writerow(
            ['TrainSize', 'TestSize', 'Run', 'Epoch', 'TrainError', 'f1', 'precision', 'recall', 'accuracy',
             'TestError', 'f1', 'precision', 'recall', 'accuracy', 'TimeStart', 'TimeEnd', 'TimePeriod'])
        for Train_size in self.TrainSizeSet:
            for Test_size in self.TestSizeSet:
                for run in range(5):
                    print(f'TRAINSIZE: {Train_size}')
                    print(f'TESTSIZE: {Test_size}')
                    print(f'Run: {run}')
                    n_train_iterations = math.ceil(Train_size / self.BatchSize)
                    n_test_iterations = math.ceil(Test_size / self.BatchSize)

                    train_loader, test_loader = self.DataLoader(Train_size, Test_size, Data)
                    model = RegularNN(self.NumNodes, self.NumWorms, self.TrainStep)
                    optimizer = torch.optim.SGD(model.parameters(), lr=self.LearningRate)

                    def train_run(model):
                        model.train()
                        for i, (inputs, outputs) in enumerate(train_loader):
                            optimizer.zero_grad()
                            predict_output = model(inputs)
                            loss, predict_error = self.Loss(predict_output, outputs)
                            loss.backward()
                            optimizer.step()

                            # self.count_parameters_table_show(model)
                            if i + 1 == n_train_iterations:
                                print(f'-----------------Train: Step {i + 1}/{n_train_iterations}, Loss {loss.item()}')
                        return loss.item(), predict_error

                    def test_run(model):
                        for i, (inputs, outputs) in enumerate(test_loader):

                            model.eval()
                            predict_output = model(inputs)
                            loss, predict_error = self.Loss(predict_output, outputs)
                            if i + 1 == n_test_iterations:
                                print(f'-----------------Test: Step {i + 1}/{n_test_iterations}, Loss {loss.item()}')
                        return loss.item(), predict_error

                    for epoch in range(self.NumEpoch):
                        print(f'Epoch: {epoch + 1}/{self.NumEpoch}')
                        t_s = time.time()
                        t_start = time.localtime()
                        time_start = time.strftime('%H%M%S', t_start)
                        train_error, predict_error = train_run(model)
                        test_error, predict_error1 = test_run(model)
                        t_e = time.time()
                        t_end = time.localtime()
                        time_end = time.strftime('%H%M%S', t_end)
                        if epoch + 1 == self.NumEpoch:
                            results_writer.writerow([Train_size,
                                                Test_size,
                                                 run,
                                                 epoch,
                                                 train_error,
                                                 predict_error[0],
                                                 predict_error[1],
                                                 predict_error[2],
                                                 predict_error[3],
                                                 test_error,
                                                 predict_error1[0],
                                                 predict_error1[1],
                                                 predict_error1[2],
                                                 predict_error1[3],
                                                 time_start,
                                                 time_end,
                                                 t_e - t_s
                                                 ])
            # results_writer.writerow(
            #     ['TrainSize', 'TestSize', 'Run', 'Epoch', 'TrainError', 'f1', 'precision', 'recall', 'accuracy',
            #      'TestError', 'f1', 'precision', 'recall', 'accuracy', 'TimeStart', 'TimeEnd', 'TimePeriod'])


class SVM_WriteResults():
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
    
    def ReadSamples(self):
        BaselineSample = SamplePreparation_SizeN(self.InputFilePath, self.SampleSize, self.NumNodes, self.NumWorms)
        input_, output_ = BaselineSample.Baseline_ReadWhoSample_Numpy()

        weight_file = os.path.join(self.InputFilePath, 'Weights.txt')
        threshold_file = os.path.join(self.InputFilePath, 'Threshold.txt')
        graph = BaselineSample.ReadGraph(threshold_file, weight_file)
        input_dict, output_dict = BaselineSample.SVM_Data_Preparation(graph, input_, output_)
        return input_dict, output_dict

    def Loss(self, predicted, true):
        x = predicted.shape
        a_1 = predicted.astype('int32')
        b_1 = true.astype('int32')
        a_1 = np.resize(a_1, (x))
        b_1 = np.resize(b_1, (x))

        f1 = f1_score(b_1, a_1, average='macro', zero_division=0)
        precision = precision_score(b_1, a_1, average='macro', zero_division=0)
        recall = recall_score(b_1, a_1, average='macro', zero_division=0)
        accuracy = accuracy_score(b_1, a_1)

        prediction_error = [f1, precision, recall, accuracy]

        return prediction_error

    def Main(self):
        torch.multiprocessing.set_start_method('spawn', force=True)
        inputs, outputs_real = self.ReadSamples()
        # print(type(inputs))
        results_file = os.path.join(self.OutputFilePath, f'SVM.csv')
        results_record = open(results_file, 'w')
        results_writer = csv.writer(results_record, delimiter=',')
        results_writer.writerow(
            ['TrainSize', 'TestSize', 'Run',
             'f1', 'precision', 'recall', 'accuracy', 'TimeStart', 'TimeEnd', 'TimePeriod'])

        for Train_size in self.TrainSizeSet:
            for Test_size in self.TestSizeSet:
                for run in range(5):
                    print(f'TRAINSIZE: {Train_size}')
                    print(f'TESTSIZE: {Test_size}')
                    print(f'Run: {run}')
                    t_s = time.time()
                    t_start = time.localtime()
                    time_start = time.strftime('%H%M%S', t_start)

                    f1 = 0
                    precision = 0
                    recall = 0
                    accuracy = 0
                    count = 0
                    for node in range(self.NumNodes):
                        if node in inputs.keys() and len(inputs[node][0]) != 0:
                            count += 1
                            node_input = inputs[node]
                            # if len(node_input) == 0:
                            #     continue
                            node_output = outputs_real[node]

                            X_train, X_test, y_train, y_test = train_test_split(node_input, node_output,
                                                                                test_size=Test_size, train_size=Train_size,
                                                                                shuffle=True
                                                                                )

                            clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
                            # if len(X_train) == 0:
                            #     # print()
                            # print(f'x', node, len(X_train[0]))
                            # print(f'y', y_train)
                            clf.fit(X_train, y_train)
                            predicted_output = clf.predict(X_test)
                            predict_error = self.Loss(predicted_output, y_test)
                            f1 += predict_error[0]
                            precision += predict_error[1]
                            recall += predict_error[2]
                            accuracy += predict_error[3]

                    t_e = time.time()
                    t_end = time.localtime()
                    time_end = time.strftime('%H%M%S', t_end)
                    results_writer.writerow([Train_size,
                                             Test_size,
                                             run,
                                             f1/count,
                                             precision/count,
                                             recall/count,
                                             accuracy/count,
                                             time_start,
                                             time_end,
                                             t_e - t_s
                                             ])
            results_writer.writerow(
                    ['TrainSize', 'TestSize', 'Run',
                     'f1', 'precision', 'recall', 'accuracy', 'TimeStart', 'TimeEnd', 'TimePeriod'])


class Random_WriteResults():
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

    def ReadSamples(self):
        BaselineSample = SamplePreparation_SizeN(self.InputFilePath, self.SampleSize, self.NumNodes, self.NumWorms)
        Data = BaselineSample.Baseline_ReadWhoSample()
        return Data
    
    def DataLoader(self, train_size, Data):
        train_data, train_rest = torch.split(Data, [train_size, self.SampleSize - train_size])
        train_dataset = WormsPropagationData_SizeN(train_data, self.NumNodes)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.BatchSize,
                                  shuffle=True,
                                  num_workers=self.NumWorkers)
        return train_loader

    def Loss(self, predicted, true):
        x, y = predicted.shape
        a_1 = predicted.cpu()
        a_1 = a_1.detach().numpy().astype('int32')
        b_1 = true.cpu()
        b_1 = b_1.detach().numpy().astype('int32')
        a_1 = np.resize(a_1, (x*y))
        b_1 = np.resize(b_1, (x*y))

        # print(b_1.shape)
        # print(a_1.shape)
        # print(f'predicted: {predicted}')
        # print(f'true{true}')

        f1 = f1_score(b_1, a_1, average='macro')
        precision = precision_score(b_1, a_1, average='macro')
        recall = recall_score(b_1, a_1, average='macro')
        accuracy = accuracy_score(b_1, a_1)

        prediction_error = [f1, precision, recall, accuracy]

        return prediction_error

    def Main(self):
        torch.multiprocessing.set_start_method('spawn', force=True)
        Data = self.ReadSamples()

        reults_file = os.path.join(self.OutputFilePath,
                                   f'Batch{self.BatchSize}Lr{self.LearningRate}Epoch{self.NumEpoch}.csv')
        results_record = open(reults_file, 'w')
        results_writer = csv.writer(results_record, delimiter=',')
        results_writer.writerow(
            ['TrainSize', 'Run', 'f1', 'precision', 'recall', 'accuracy', 'TimeStart', 'TimeEnd', 'TimePeriod'])
        for Train_size in self.TrainSizeSet:
            for run in range(5):
                print(f'TRAINSIZE: {Train_size}')
                print(f'Run: run')
                train_loader = self.DataLoader(Train_size, Data)
                def train_run():
                    for i, (inputs, outputs) in enumerate(train_loader):
                        # Run training process
                        # predict_output = model(inputs)
                        a, b = inputs.shape
                        predict_output = torch.zeros(a, b)
                        for i in range(a):
                            for j in range(b):
                                predict_output[i, j] = random.randint(0, self.NumWorms)

                        predict_error = self.Loss(predict_output, outputs)
                        return predict_error
                t_s = time.time()
                t_start = time.localtime()
                time_start = time.strftime('%H%M%S', t_start)
                predict_error = train_run()
                t_e = time.time()
                t_end = time.localtime()
                time_end = time.strftime('%H%M%S', t_end)
                results_writer.writerow([Train_size,
                                         run,
                                         predict_error[0],
                                         predict_error[1],
                                         predict_error[2],
                                         predict_error[3],
                                         time_start,
                                         time_end,
                                         t_e - t_s
                                         ])
            results_writer.writerow(
                ['TrainSize', 'Run', 'f1', 'precision', 'recall', 'accuracy', 'TimeStart', 'TimeEnd', 'TimePeriod'])
