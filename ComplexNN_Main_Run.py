from Tools.ComplexNNs_Tools import *
from Real_graph_data import intel_lab_data


# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


Graph_Name = 'ERgraph'
NumNode = 200
NumSeed = 50
NumWorms = 4
SampleSize = 1000

TrainStep = 10
Path = os.getcwd()

InputFilePath = os.path.join(Path, f'Samples/{Graph_Name}/Samples{SampleSize}Node{NumNode}_Seed{NumSeed}_Worms{NumWorms}')
OutputFilePath = os.path.join(Path, f'Results/{Graph_Name}/ComplexNN/Samples{SampleSize}Node{NumNode}_Seed{NumSeed}_Worms{NumWorms}_TrainStep{TrainStep}')


if os.path.exists(OutputFilePath) == False:
    os.mkdir(OutputFilePath)

TrainSizeSet = [600]
TestSizeSet = [400]

BatchSize = 100
LearningRate = 1
NumEpoch = 40


NumWorkers = 0
torch.set_printoptions(precision=5)
ComplexNNTrain = MainRun_WriteResults(OutputFilePath,
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


ComplexNNTrain.Main()
