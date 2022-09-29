from Tools.BaselineMethod_Tools import *
import os
Graph_Name = 'Powerlaw'
NumNode = 768
NumSeed = 300
NumWorms = 4
SampleSize = 1000

TrainStep = 10
InputFilePath = f'/home/cds/Documents/worms_propagation_model/Samples/{Graph_Name}/Samples{SampleSize}Node{NumNode}_Seed{NumSeed}_Worms{NumWorms}'
OutputFilePath = f'/home/cds/Documents/worms_propagation_model/Results/{Graph_Name}/SVM/Samples{SampleSize}Node{NumNode}_Seed{NumSeed}_Worms{NumWorms}'

if os.path.exists(OutputFilePath) == False:
    os.mkdir(OutputFilePath)

TrainSizeSet = [100, 200, 400, 600]
TestSizeSet = [400]

BatchSize = 100
LearningRate = 1
NumEpoch = 40


NumWorkers = 0
torch.set_printoptions(precision=5)
SVMTrain = SVM_WriteResults(OutputFilePath,
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


SVMTrain.Main()
