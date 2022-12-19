'''
Created on Oct. 8, 2022

@author: Shalom
'''

import matplotlib.pyplot as plt
import numpy as np
import math
import random

##############File I/O#################

input_file = open('dataset1_inputs.txt', 'r')
output_file = open('dataset1_outputs.txt', 'r')

inputs = []
outputs = []

inputdata = input_file.readlines()
outputdata = output_file.readlines()

for line in inputdata:
    inputs.append(float(line.strip()))
    
for line in outputdata:
    outputs.append(float(line.strip()))

inputMap = dict(zip(inputs, outputs))
print(inputMap)

print(inputs)
print(outputs)

input_file.close()
output_file.close() 

##############STEP 1: LOADING DATA#################


plt.plot(inputs, outputs, 'o', markersize=5, color='blue')
plt.xlabel('Inputs')
plt.ylabel('Outputs')
plt.title('Data Plot')
plt.show()


##################STEP 2: ERM#####################

ERMLoss = [] #array that stores the empirical square loss for each polynomial of degree W. ERM[i] = empirical square loss for polynomial of degree W=i+1

for degree in range (1, 21): #for every degree between 1 and 20
    
    DM = [] #design matrix for each degree
    for N in inputs:
        LinearFunction = [] #linear function of degree for input N
        for D in range (0, degree+1):
            LinearFunction.append(pow(N, D))
        DM.append(LinearFunction)

    print(DM)
    
    DMTranspose = np.transpose(DM)
    W = np.linalg.solve(np.dot(DMTranspose, DM), np.dot(DMTranspose, outputs))
    a = np.subtract(np.dot(DM,W), outputs)
    
    ERMLoss.append((1/len(inputs)) * (1/2) * pow(np.linalg.norm(a),2))
    

degrees = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20];

plt.plot(degrees, ERMLoss, 'o-', color='blue')
plt.yticks(np.arange(0, 150, 10))
plt.xticks(np.arange(0, 21, 1))

plt.xlabel('Degree W')
plt.ylabel('ESL for ERM')
plt.title('Empirical Square Loss (ESL) by Degree for ERM Model')
plt.show()

#######################STEP 3: RLM############################

RLMLoss = []
DM20 = [] #design matrix for degree 20


for N in inputs:
    LinearFunction = [] #linear function of degree 20 for input N
    for D in range (0, 21):
        LinearFunction.append(pow(N, D))
    DM20.append(LinearFunction)

DM20Transpose = np.transpose(DM20)

for i in range(-10, 10):
    regularizer = pow(math.e, i)
    W = np.linalg.solve(np.add(np.dot(DM20Transpose, DM20), np.dot(regularizer, np.identity(21))), np.dot(DM20Transpose, outputs))
    a = np.subtract(np.dot(DM20,W), outputs)
    RLMLoss.append((1/len(inputs)) * (1/2) * pow(np.linalg.norm(a),2) + (regularizer/2)*pow(np.linalg.norm(W),2))

print(RLMLoss)
i = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9]

plt.plot(i, RLMLoss, 'o-', markersize=5, color='blue')
plt.yticks(np.arange(0, 200, 10))
plt.xticks(np.arange(-10, 10, 1))
plt.xlabel('i value')
plt.ylabel('ESL for RLM')
plt.title('Empirical Square Loss (ESL) by i-value for RLM Model')
plt.show()


###################STEP 4: CROSS VALIDATION###################

randomInputs = random.sample(inputs, 100)
InputChunks = []

#####split the data into 10 chunks#####
i = 0
for chunk in range(0,10):
    chunk = []
    chunk.append(randomInputs[i])
    i = i + 1
    while i%10 != 0:
        chunk.append(randomInputs[i])
        i = i + 1
    InputChunks.append(chunk)
###########for every degree###############

print(InputChunks)

AvgLossPerDegree = [] #average loss for every degree

for degree in degrees:
    TotalChunkLoss = 0

    for testInput in InputChunks: #for each chunk, select it as test set
        trainInput = [] #use everything else as train set
        trainOutput = []
        testOutput = []

        for key in inputMap:
            if key not in chunk:
                trainInput.append(key)
                trainOutput.append(inputMap.get(key))
            if key in chunk:
                testOutput.append(inputMap.get(key))
        
        DMatrix = [] #design matrix for each degree
        for N in trainInput:
            LinearFunction = [] #linear function of degree for input N
            for D in range (0, degree+1):
                LinearFunction.append(pow(N, D))
            DMatrix.append(LinearFunction)
        
        DMatrixTranspose = np.transpose(DMatrix)
        Weights = np.linalg.solve(np.dot(DMatrixTranspose, DMatrix), np.dot(DMatrixTranspose, trainOutput))
        
        #retrieve the test outputs from the ERM for the test inputs
        ERMTestOutputs = []
        for values in testInput:
            ERMy = 0
            for D in range(0, degree+1):
                ERMy = ERMy + Weights[D]*pow(values, D)
            ERMTestOutputs.append(ERMy)
        
        a = np.subtract(testOutput, ERMTestOutputs)
        TotalChunkLoss += (1/len(testOutput)) * (1/2) * pow(np.linalg.norm(a),2)
        
    AvgLossPerDegree.append(TotalChunkLoss/10)
    print(AvgLossPerDegree)

print(AvgLossPerDegree)
plt.plot(degrees, AvgLossPerDegree, 'o-', color='blue', markersize=5)

plt.xticks(np.arange(0, 21, 1))
plt.xlabel('Degree')
plt.ylabel('Average Square Loss')
plt.title('Cross-Validation')
plt.show()

###################STEP 5: VISUALIZATION######################

degrees = [1,5,10,20]

####ERM VISUALIZATION####
plt.plot(inputs, outputs, 'o', color='slateblue', label = 'data', markersize=5)
colors = ['red', 'orange', 'green', 'purple', 'black']
c = 0

for degree in degrees:
    DM = [] #design matrix for each degree
    for N in inputs:
        LinearFunction = [] #linear function of degree for input N
        for D in range (0, degree+1):
            LinearFunction.append(pow(N, D))
        DM.append(LinearFunction)

    DMTranspose = np.transpose(DM)
    
    W = np.linalg.solve(np.dot(DMTranspose, DM), np.dot(DMTranspose, outputs))
        
    ERMoutputs = []
    ERMInputs = []
    XRange = np.arange(-9.85, 10.01, 0.01)
    
    for x in XRange:
        ERMInputs.append(x)
        ERMy = 0
        for D in range(0, degree+1):
            ERMy = ERMy + W[D]*pow(x, D)
        ERMoutputs.append(ERMy)
    plt.plot(ERMInputs, ERMoutputs, '-', color = colors[c], label = 'Degree ' + str(degree))
    c = c + 1
    
plt.yticks(np.arange(-50, 101, 10))
plt.xticks(np.arange(-10, 11, 1))
plt.legend()
plt.xlabel('Input values')
plt.ylabel('ERM output values')
plt.title('ERM Learned Model Plot')
plt.show()

####RLM VISUALIZATIOn####
plt.plot(inputs, outputs, 'o', color='blue', label = 'data', markersize=5)
c = 0

for degree in degrees:
    DM = [] #design matrix for each degree
    for N in inputs:
        LinearFunction = [] #linear function of degree for input N
        for D in range (0, degree+1):
            LinearFunction.append(pow(N, D))
        DM.append(LinearFunction)

    DMTranspose = np.transpose(DM)
    
    W = np.linalg.solve(np.add(np.dot(DMTranspose, DM), np.dot(pow(math.e, 5), np.identity(degree+1))), np.dot(DMTranspose, outputs))
        
    RLMInputs = []
    RLMOutputs = []
    XRange = np.arange(-9.85, 10.01, 0.01)
    
    for x in XRange:
        RLMInputs.append(x)
        RLMy = 0
        for D in range(0, degree+1):
            RLMy = RLMy + W[D]*pow(x, D)
        RLMOutputs.append(RLMy)
    
    col = (random.random(), random.random(), random.random())
    plt.plot(RLMInputs, RLMOutputs, '-', color=colors[c], label = 'Degree ' + str(degree))
    c = c + 1
    
plt.yticks(np.arange(-50, 101, 10))
plt.xticks(np.arange(-10, 11, 1))
plt.legend()
plt.xlabel('input value')
plt.ylabel('RLM output for degree ' + str(degree))
plt.title('RLM Learned Model Plot')
plt.show()




