from Tools.BaselineMethod_Tools import *
import os

Graph_Name = 'intel_lab_data'
NumNode = 55
NumSeed = 25
NumWorms = 4
SampleSize = 1000

TrainStep = 10

Path = os.getcwd()

InputFilePath = os.path.join(Path, f'Samples/{Graph_Name}/Samples{SampleSize}Node{NumNode}_Seed{NumSeed}_Worms{NumWorms}')
OutputFilePath = os.path.join(Path, f'Results/{Graph_Name}/RegularNN/Samples{SampleSize}Node{NumNode}_Seed{NumSeed}_Worms{NumWorms}')



if os.path.exists(OutputFilePath) == False:
    os.mkdir(OutputFilePath)

TrainSizeSet = [200, 400, 600]
TestSizeSet = [400]

BatchSize = 100
LearningRate = 1
NumEpoch = 40


NumWorkers = 0
torch.set_printoptions(precision=5)
RegularNNTrain = RegularNN_WriteResults(OutputFilePath,
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
)


RegularNNTrain.Main()
