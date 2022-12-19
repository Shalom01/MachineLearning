'''
Created on Oct. 30, 2022

@author: Shalom Asbell

'''

import matplotlib.pyplot as plt
import numpy as np
import copy
from pickle import NONE
from matplotlib.ticker import MaxNLocator

##############File I/O#################

input_file = open('6pointsinputs.txt', 'r')
output_file = open('6pointsoutputs.txt', 'r')
fg_input_file = open('fg_inputs.txt', 'r')
fg_output_file = open('fg_outputs.txt', 'r')
irisnum_file = open('irisnum.txt', 'r')


inputs = []
outputs = []
fg_inputs = []
fg_outputs = []
irisnum_inputs = []
irisnum_outputs = []

inputdata = input_file.readlines()
outputdata = output_file.readlines()
fg_inputdata = fg_input_file.readlines()
fg_outputdata = fg_output_file.readlines()
irisnum_data = irisnum_file.readlines()

for line in inputdata:
    line = line.strip()
    point = line.split(", ")
    point = [float(number) for number in point]
    inputs.append(point)

for line in fg_inputdata:
    line = line.strip()
    point = line.split(",")
    point = [float(number) for number in point]
    fg_inputs.append(point)

for line in irisnum_data:
    line = line.strip()
    point = line.split(",")
    irisnum_outputs.append(float(point[len(point)- 1]))
    del(point[len(point)- 1])
    point = [float(number) for number in point]
    irisnum_inputs.append(point)    

for line in outputdata:
    outputs.append(float(line.strip()))
    
for line in fg_outputdata:
    fg_outputs.append(float(line.strip()))

inputMap = {}
for i in range(0, len(inputs)):
    inputMap[tuple(inputs[i])] = outputs[i]
    
input_file.close()
output_file.close()
fg_input_file.close()
fg_output_file.close()

##############STEP 1: LOADING DATA AND ANALYZE FIRST DATASET####################################################################################
print("---------------------------------------------------------------------------------------------------------------------------------------")
print("\033[1m" + "STEP 1 - LOADING DATA AND ANALYZE FIRST DATASET:" + "\033[0m")
print("See plotted graph for plot of 6points data set.")

#below I implement a simple plot for the data#

for i in inputs:
    if inputMap[tuple(i)] == 1.0:
        plt.plot(i[0], i[1], '+', markersize=10, mew=2, color='red')
    else:
        plt.plot(i[0], i[1], '_', markersize=10, mew=2, color='blue')

axis = plt.gca()
axis.spines['top'].set_position('zero')
axis.spines['left'].set_position('zero')
axis.spines['right'].set_position('zero')
axis.spines['bottom'].set_position('zero')
axis.spines['top'].set_color('black')

axis.xaxis.set_major_locator(MaxNLocator(integer=True))
axis.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.xlabel('X\u2081')
plt.ylabel('X\u2082')
plt.title('6 Points Data Plot')
plt.grid(True)
plt.show()

##############STEP 2: LDA AND FISHER############################################################################################################
print("---------------------------------------------------------------------------------------------------------------------------------------")
print("\033[1m" + "STEP 2 - FISHER CRITERION FOR EACH OUTPUT:" + "\033[0m")

#In the code below I calculate the Fisher criterion for all w vectors calculated in STEP 1#

Mpositive = [0,0] #mean vector for output 1
Mnegative = [0,0] #mean vector for output -1
Cpositive = 0 #number of positive outputs
Cnegative = 0 #number of negative outputs

for i in inputs:
    if inputMap[i[0], i[1]]==1.0:
        Mpositive = np.add(Mpositive, i)
        Cpositive+=1
    else:
        Mnegative = np.add(Mnegative, i)
        Cnegative+=1

Mpositive = np.divide(Mpositive, Cnegative)
Mnegative = np.divide(Mnegative, Cpositive)

w1 = [1, -1] #w vector derived from Perceptron with (-1, 1) first or (1, -1) first.
w2 = [2, -1] #w vector derived from Perceptron with (2, -1) first.
w3 = [3, -1] #w vector derived from Perceptron with (3, -1) first.
w4 = [1, -2] #w vector derived from Perceptron with (-1, 2) first.
w5 = [1, -3] #w vector derived from Perceptron with (-1, 3) first.

MpositiveW1 = np.dot(Mpositive, np.transpose(w1))
MnegativeW1 = np.dot(Mnegative, np.transpose(w1))

MpositiveW2 = np.dot(Mpositive, np.transpose(w2))
MnegativeW2 = np.dot(Mnegative, np.transpose(w2))

MpositiveW3 = np.dot(Mpositive, np.transpose(w3))
MnegativeW3 = np.dot(Mnegative, np.transpose(w3))

MpositiveW4 = np.dot(Mpositive, np.transpose(w4))
MnegativeW4 = np.dot(Mnegative, np.transpose(w4))

MpositiveW5 = np.dot(Mpositive, np.transpose(w5))
MnegativeW5 = np.dot(Mnegative, np.transpose(w5))


SpositiveVarianceW1 = 0
SnegativeVarianceW1 = 0

SpositiveVarianceW2 = 0
SnegativeVarianceW2 = 0

SpositiveVarianceW3 = 0
SnegativeVarianceW3 = 0

SpositiveVarianceW4 = 0
SnegativeVarianceW4 = 0

SpositiveVarianceW5 = 0
SnegativeVarianceW5 = 0

for i in inputs:
    if inputMap[i[0], i[1]] == 1.0:        
        SpositiveVarianceW1 += np.square(np.dot(np.transpose(w1), i) - MpositiveW1)
        SpositiveVarianceW2 += np.square(np.dot(np.transpose(w2), i) - MpositiveW2)
        SpositiveVarianceW3 += np.square(np.dot(np.transpose(w3), i) - MpositiveW3)
        SpositiveVarianceW4 += np.square(np.dot(np.transpose(w4), i) - MpositiveW4)
        SpositiveVarianceW5 += np.square(np.dot(np.transpose(w5), i) - MpositiveW5)
    else:
        SnegativeVarianceW1 += np.square(np.dot(np.transpose(w1), i) - MnegativeW1)
        SnegativeVarianceW2 += np.square(np.dot(np.transpose(w2), i) - MnegativeW2)
        SnegativeVarianceW3 += np.square(np.dot(np.transpose(w3), i) - MnegativeW3)
        SnegativeVarianceW4 += np.square(np.dot(np.transpose(w4), i) - MnegativeW4)
        SnegativeVarianceW5 += np.square(np.dot(np.transpose(w5), i) - MnegativeW5)
        
        
FisherW1 = np.divide((np.square(MpositiveW1 - MnegativeW1)), (SpositiveVarianceW1 + SnegativeVarianceW1))
FisherW2 = np.divide((np.square(MpositiveW2 - MnegativeW2)), (SpositiveVarianceW2 + SnegativeVarianceW2))
FisherW3 = np.divide((np.square(MpositiveW3 - MnegativeW3)), (SpositiveVarianceW3 + SnegativeVarianceW3))
FisherW4 = np.divide((np.square(MpositiveW4 - MnegativeW4)), (SpositiveVarianceW4 + SnegativeVarianceW4))
FisherW5 = np.divide((np.square(MpositiveW5 - MnegativeW5)), (SpositiveVarianceW5 + SnegativeVarianceW5))

print("w = [1, -1] has Fisher = " + str(FisherW1))
print("w = [2, -1] has Fisher = " + str(FisherW2))
print("w = [3, -1] has Fisher = " + str(FisherW3))
print("w = [1, -2] has Fisher = " + str(FisherW4))
print("w = [1, -3] has Fisher = " + str(FisherW5))

##############STEP 3: IMPLEMENT PERCEPTRON######################################################################################################
print("---------------------------------------------------------------------------------------------------------------------------------------")
print("\033[1m" + "STEP 3 - IMPLEMENT PERCEPTRON:" + "\033[0m")

###############################################################################################################
#Class PerceptronCalculator has functions:                                                                    #  
#Perceptron(inputs, outputs) - inputs an input vector of data, outputs are labels of data in {-1, 1}.         #
#note that inputs have to correctly ordered with their outputs (bijection between input[i] and output[i]).    #
#precondition: data must be linearly separable. Returns W vector that linearly separates data.                #
#allCorrectlyClassified() - helper method for Perceptron(), returns NONE if all correctly classified, or      #
#will return vector of incorrectly classified input.                                                          #
#getStatistics() - returns a String with statistics about the last perceptron run. Precondition: Perceptron() #
#must have been run at least once.                                                                            #
###############################################################################################################

class PerceptronCalculator:
    
    passes = 0 #variable to store the number of passes over the dataset
    inputCopy = [] #class copy of input
    outputCopy = [] #class copy of output
    mapInput = {} #dictionary mapping of inputs to outputs
    w = [] #vector W that stores the W calculated from last Perceptron run (initially empty)
    
    ###############################################################################################
    #function that runs perceptron on dataset with inputs (of dimension D) and outputs with labels#
    # {1, -1}. Returns homogeneous linear classifier a W of degree (D + 1).                       #
    # precondition: data must be linearly seperable                                               #
    ###############################################################################################

    def Perceptron(self, inputs, outputs):
    
        self.inputCopy = copy.deepcopy(inputs) #deepcopy of inputs
        self.outputCopy = copy.deepcopy(outputs)
        
        for i in self.inputCopy: #append 1 to each input in inputs for an input size of D+1
            i.insert(0, 1.0)
               
        for i in range(0, len(self.inputCopy)): #create inputMap with inputs of size D+1
            self.mapInput[tuple(self.inputCopy[i])] = self.outputCopy[i]
    
        self.w = []
        for i in range(0, len(self.inputCopy[0])): #create empty W vector of size D+1
            self.w.append(0)    
        
        np.random.shuffle(self.inputCopy)
        self.passes = 0
        classified = False
        
        while self.passes < 100 and not classified:
            i = self._allCorrectlyClassified(self.w, self.inputCopy) #check if all points are correctly classified
            
            if i is NONE: #if all are correctly classified
                classified = True
            else: #if a point is not correctly classified
                self.w = np.add(self.w, np.multiply(self.mapInput.get(tuple(i)), i)) #update the W vector appropriately
            self.passes+=1
            
        self.w = np.divide(self.w, np.linalg.norm(self.w)) #normalize the W vector
        return self.w

    def _allCorrectlyClassified(self, w, inputs): #helper function that returns the input that is not correctly classified. If all correctly classified, returns null.
        for i in inputs:
            output = self.mapInput.get(tuple(i))        
            if (output * np.dot(w, i)) <= 0:
                return i 
        return NONE #all were correctly classified
    
    def getStatistics(self): #returns a string with input, update and W information
        return "Perceptron run on inputs (D-dimensional):" + str(inputs) + "\nNumber of updates of W: " + str(self.passes - 1) + "\nNormalized output vector W (D+1-dimensional): " + str(self.w)
    
perceptronCalculator = PerceptronCalculator()


for i in range(1, 11):
    perceptronCalculator.Perceptron(inputs, outputs)
    print("\033[4m" + "6 Data Points Perceptron Run " + str(i) + " Summary " + "\033[0m")
    print(perceptronCalculator.getStatistics())


#############STEP 4: IMPLEMENT 3 LOSS FUNCTIONS#################################################################################################
print("---------------------------------------------------------------------------------------------------------------------------------------")
print("\033[1m" + "STEP 4 - IMPLEMENT 3 LOSS FUNCTIONS:" + "\033[0m")

##############################################################################################
#function that returns the binary loss, hinge loss, and logistic loss                        #
#for a given dataset with inputs (D dimension) and outputs and a given W (of D + 1 dimension)#
##############################################################################################
def computeLoss(inputs, outputs, w):
    
    input3D = copy.deepcopy(inputs)
    outputs3D = copy.deepcopy(outputs)
    
    for i in input3D: #D+1 vector of inputs with 1 appended to beginning of inputs
        i.insert(0, 1.0)
        
    BL = 0
    HL = 0
    LL = 0

    for i in range(0, len(input3D)):  
        BL += binaryLoss(w, input3D[i], outputs3D[i])
        HL += hingeLoss(w, input3D[i], outputs3D[i])
        LL += logisticLoss(w, input3D[i], outputs3D[i])

    return BL/len(input3D), HL/len(input3D), LL/len(input3D)
    
def binaryLoss(w, x, t): #helper function, returns the binary loss of a give W (D + 1), X (D + 1) and label t element of {1, -1}
    loss = 0
    if np.sign(np.dot(w, np.transpose(x))) != t:
        loss = 1
    return loss

def hingeLoss(w, x, t): #helper function, returns the hinge loss of a give W (D + 1), X (D + 1) and label t element of {1, -1}
    return max(0, (1 - (t * np.dot(w, np.transpose(x)))))
        
def logisticLoss(w, x, t): #helper function, returns the logistic loss of a give W (D + 1), X (D + 1) and label t element of {1, -1}
        return -np.log(1.0/(1 + np.exp(-(t * np.dot(w, np.transpose(x))))))

######lets run the perceptron 20 times on the 6 point data set and compute the losses########

W20 = [] #array to hold the W vectors that were output from the 20 runs

for r in range(0, 20):
    W20.append(perceptronCalculator.Perceptron(inputs, outputs)) #run the perceptron on the inputs 20 times (note we shuffle the inputs in the perceptron algorithm)    
    
BL = [0]*20 #array to hold the average binary loss over the data points for each of the 20 W's
HL = [0]*20 #array to hold the average hinge loss over the data points for each of the 20 W's
LL = [0]*20 #array to hold the average logistic loss over the data points for each of the 20 W's

for r in range(0, 20):
    BL[r] = computeLoss(inputs, outputs, W20[r])[0]
    HL[r] = computeLoss(inputs, outputs, W20[r])[1]
    LL[r] = computeLoss(inputs, outputs, W20[r])[2]
    
print("\033[2m" + "See plotted graph for loss summary of 20 runs on inputs." + "\033[2m")

plt.plot(list(range(1, 21)), BL, '-', color='orange', label = 'Binary Loss')
plt.plot(list(range(1, 21)), HL, '-', color='blue', label = 'Hinge Loss')
plt.plot(list(range(1, 21)), LL, '-', color='green', label = 'Logistic Loss')
plt.xticks(np.arange(0, 21, 1))
plt.xlabel("Trial Number")
plt.ylabel("Average Loss")
plt.legend(loc = 'upper right')
plt.title("Loss Summary of 20 Perceptron Runs on 6 Inputs")
plt.grid()
plt.show()

######lets scale the inputs by a factor 10  the perceptron 20 times on the 6 point data set and compute the losses########
scaledinputs = []

for i in inputs:
    scaledinputs.append(np.multiply(i, 10).tolist())

W20scaled = [] #array to hold the W vectors that were output from the 20 runs

for r in range(0, 20):
    W20scaled.append(perceptronCalculator.Perceptron(scaledinputs, outputs)) #run the perceptron on the inputs 20 times (note we shuffle the inputs in the perceptron algorithm)

BLscaled = [0]*20 #array to hold the average binary loss over the data points for each of the 20 W's
HLscaled = [0]*20 #array to hold the average hinge loss over the data points for each of the 20 W's
LLscaled = [0]*20 #array to hold the average logistic loss over the data points for each of the 20 W's

for r in range(0, 20):
    BLscaled[r] = computeLoss(scaledinputs, outputs, W20scaled[r])[0]
    HLscaled[r] = computeLoss(scaledinputs, outputs, W20scaled[r])[1]
    LLscaled[r] = computeLoss(scaledinputs, outputs, W20scaled[r])[2]
    
print("\033[2m" + "See plotted graph for loss summary of 20 runs on scaled inputs." + "\033[2m")

plt.plot(list(range(1, 21)), BLscaled, '-', linewidth = 1.5, color='orange', label = 'Binary Loss')
plt.plot(list(range(1, 21)), HLscaled, ':', linewidth = 1.5, color='blue', label = 'Hinge Loss')
plt.plot(list(range(1, 21)), LLscaled, '-', linewidth = 1.5, color='green', label = 'Logistic Loss')
plt.xticks(np.arange(0, 21, 1))
plt.xlabel("Trial Number")
plt.ylabel("Average Loss")
plt.legend(loc = 'upper right')
plt.title("Loss Summary of 20 Perceptron Runs on Scaled 6 Inputs")
plt.grid()
plt.show()

#############STEP 5: COMPARE LOSS FUNCTIONS#####################################################################################################

print("---------------------------------------------------------------------------------------------------------------------------------------")
print("\033[1m" + "STEP 5 - COMPARE LOSS FUNCTIONS:" + "\033[0m")

W20fg = [] #array to hold the W vectors that were output from the 20 runs

for r in range(0, 20):
    W20fg.append(perceptronCalculator.Perceptron(fg_inputs, fg_outputs)) #run the perceptron on the inputs 20 times (note we shuffle the inputs in the perceptron algorithm)

BLfg = [0]*20 #array to hold the average binary loss over the data points for each of the 20 W's
HLfg = [0]*20 #array to hold the average hinge loss over the data points for each of the 20 W's
LLfg = [0]*20 #array to hold the average logistic loss over the data points for each of the 20 W's

BLfg[0] = computeLoss(fg_inputs, fg_outputs, W20fg[0])[0]
HLfg[0] = computeLoss(fg_inputs, fg_outputs, W20fg[0])[1]
LLfg[0] = computeLoss(fg_inputs, fg_outputs, W20fg[0])[2]

SmallestBL = BLfg[0] #smallest average BL
SmallestHL = HLfg[0] #smallest average HL
SmallestLL = LLfg[0] #smallest average LL

SmallestBLW = W20fg[0] #vector with smallest average BL
SmallestHLW = W20fg[0] #vector with smallest average HL
SmallestLLW = W20fg[0] #vector with smallest average LL

SmallestBLRun = 1;
SmallestHLRun = 1;
SmallestLLRun = 1;

for r in range(1, 20):
    BLfg[r] = computeLoss(fg_inputs, fg_outputs, W20fg[r])[0]
    HLfg[r] = computeLoss(fg_inputs, fg_outputs, W20fg[r])[1]
    LLfg[r] = computeLoss(fg_inputs, fg_outputs, W20fg[r])[2]
    
    if BLfg[r] < SmallestBL:
        SmallestBL = BLfg[r]
        SmallestBLW = W20fg[r]
        SmallestBLRun = r + 1;
        
    if HLfg[r] < SmallestHL:
        SmallestHL = HLfg[r] 
        SmallestHLW = W20fg[r]
        SmallestHLRun = r + 1;

        
    if LLfg[r] < SmallestLL:
        SmallestLL = LLfg[r]
        SmallestLLW = W20fg[r]
        SmallestLLRun = r + 1;
    
    
print("See plotted graph for loss summary of 20 runs on fg data.")

print("\033[4m" + "Vector W that minimizes losses:" + "\033[0m")

print("W that minimizes binary loss: " + str(SmallestBLW) + " (run " + str(SmallestBLRun) + ")")
print("W that minimizes hinge loss: " + str(SmallestHLW) + " (run " + str(SmallestHLRun) + ")")
print("W that minimizes logistic loss: " + str(SmallestLLW) + " (run " + str(SmallestLLRun) + ")")

plt.plot(list(range(1, 21)), BLfg, '-', linewidth = 1.5, color='orange', label = 'Binary Loss')
plt.plot(list(range(1, 21)), HLfg, ':', linewidth = 1.5, color='blue', label = 'Hinge Loss')
plt.plot(list(range(1, 21)), LLfg, '-', linewidth = 1.5, color='green', label = 'Logistic Loss')
plt.xticks(np.arange(0, 21, 1))
plt.xlabel("Trial Number")
plt.ylabel("Average Loss")
plt.legend(loc = 'upper right')
plt.legend()
plt.title("Loss Summary for 20 Runs on fg Inputs")
plt.grid()
plt.show()

#############STEP 6: MULTICLASS################################################################################################################
print("---------------------------------------------------------------------------------------------------------------------------------------")
print("\033[1m" + "STEP 6 - MULTICLASS:" + "\033[0m")

#train linear classifier 1 (1 versus 2, 3 classifier)

W20iris1 = [] #array to hold the W vectors that were output from the 20 runs
Classifier1Outputs = copy.deepcopy(irisnum_outputs)


for i in range(0, len(Classifier1Outputs)):
    if Classifier1Outputs[i] !=  1:
        Classifier1Outputs[i] = -1.0        

for r in range(0, 20):
    W20iris1.append(perceptronCalculator.Perceptron(irisnum_inputs, Classifier1Outputs)) #run the perceptron on the inputs 20 times (note we shuffle the inputs in the perceptron algorithm)

Classifier1BL = [0]*20 #array to hold the average binary loss over the data points for each of the 20 W's
Classifier1HL = [0]*20 #array to hold the average hinge loss over the data points for each of the 20 W's
Classifier1LL = [0]*20 #array to hold the average logistic loss over the data points for each of the 20 W's

for r in range(0, 20):
    Classifier1BL[r] = computeLoss(irisnum_inputs, Classifier1Outputs, W20iris1[r])[0]
    Classifier1HL[r] = computeLoss(irisnum_inputs, Classifier1Outputs, W20iris1[r])[1]
    Classifier1LL[r] = computeLoss(irisnum_inputs, Classifier1Outputs, W20iris1[r])[2]
    
#train linear classifier 2 (2 versus 1, 3 classifier)

W20iris2 = [] #array to hold the W vectors that were output from the 20 runs
Classifier2Outputs = copy.deepcopy(irisnum_outputs)

for i in range(0, len(Classifier2Outputs)):
    if Classifier2Outputs[i] !=  2:
        Classifier2Outputs[i] = -1.0
    else:
        Classifier2Outputs[i] = 1.0      

for r in range(0, 20):
    W20iris2.append(perceptronCalculator.Perceptron(irisnum_inputs, Classifier2Outputs)) #run the perceptron on the inputs 20 times (note we shuffle the inputs in the perceptron algorithm)

Classifier2BL = [0]*20 #array to hold the average binary loss over the data points for each of the 20 W's
Classifier2HL = [0]*20 #array to hold the average hinge loss over the data points for each of the 20 W's
Classifier2LL = [0]*20 #array to hold the average logistic loss over the data points for each of the 20 W's

for r in range(0, 20):
    Classifier2BL[r] = computeLoss(irisnum_inputs, Classifier2Outputs, W20iris2[r])[0]
    Classifier2HL[r] = computeLoss(irisnum_inputs, Classifier2Outputs, W20iris2[r])[1]
    Classifier2LL[r] = computeLoss(irisnum_inputs, Classifier2Outputs, W20iris2[r])[2]

#train linear classifier 3 (3 versus 1, 2 classifier)

W20iris3 = [] #array to hold the W vectors that were output from the 20 runs
Classifier3Outputs = copy.deepcopy(irisnum_outputs)

for i in range(0, len(Classifier3Outputs)):
    if Classifier3Outputs[i] !=  3:
        Classifier3Outputs[i] = -1.0
    else:
        Classifier3Outputs[i] = 1.0

for r in range(0, 20):
    W20iris3.append(perceptronCalculator.Perceptron(irisnum_inputs, Classifier3Outputs)) #run the perceptron on the inputs 20 times (note we shuffle the inputs in the perceptron algorithm)

Classifier3BL = [0]*20 #array to hold the average binary loss over the data points for each of the 20 W's
Classifier3HL = [0]*20 #array to hold the average hinge loss over the data points for each of the 20 W's
Classifier3LL = [0]*20 #array to hold the average logistic loss over the data points for each of the 20 W's

for r in range(0, 20):
    Classifier3BL[r] = computeLoss(irisnum_inputs, Classifier3Outputs, W20iris3[r])[0]
    Classifier3HL[r] = computeLoss(irisnum_inputs, Classifier3Outputs, W20iris3[r])[1]
    Classifier3LL[r] = computeLoss(irisnum_inputs, Classifier3Outputs, W20iris3[r])[2]

###print the loss summaries and display the loss plots for each "one versus all" classifier###

print("See plotted graph for 1 versus {2,3} classifier losses.")

plt.plot(list(range(1, 21)), Classifier1BL, '-', linewidth = 1.5, color='orange', label = 'Binary Loss')
plt.plot(list(range(1, 21)), Classifier1HL, ':', linewidth = 1.5, color='blue', label = 'Hinge Loss')
plt.plot(list(range(1, 21)), Classifier1LL, '-', linewidth = 1.5, color='green', label = 'Logistic Loss')
plt.xticks(np.arange(0, 21, 1))
plt.xlabel("Trial Number")
plt.ylabel("Average Loss")
plt.legend(loc = 'upper right')
plt.title("1 versus {2,3} classifier loss summary")
plt.legend()
plt.grid()
plt.show()


print("See plotted graph for 2 versus {1,3} classifier losses.")

plt.plot(list(range(1, 21)), Classifier2BL, '-', linewidth = 1.5, color='orange', label = 'Binary Loss')
plt.plot(list(range(1, 21)), Classifier2HL, ':', linewidth = 1.5, color='blue', label = 'Hinge Loss')
plt.plot(list(range(1, 21)), Classifier2LL, '-', linewidth = 1.5, color='green', label = 'Logistic Loss')
plt.xticks(np.arange(0, 21, 1))
plt.xlabel("Trial Number")
plt.ylabel("Average Loss")
plt.legend(loc = 'upper right')
plt.title("2 versus {1,3} classifier loss summary")
plt.legend()
plt.grid()
plt.show()

print("See plotted graph for 3 versus {1,2} classifier losses.")

plt.plot(list(range(1, 21)), Classifier3BL, '-', linewidth = 1.5, color='orange', label = 'Binary Loss')
plt.plot(list(range(1, 21)), Classifier3HL, ':', linewidth = 1.5, color='blue', label = 'Hinge Loss')
plt.plot(list(range(1, 21)), Classifier3LL, '-', linewidth = 1.5, color='green', label = 'Logistic Loss')
plt.xticks(np.arange(0, 21, 1))
plt.xlabel("Trial Number")
plt.ylabel("Average Loss")
plt.legend(loc = 'upper right')
plt.title("3 versus {1,2} classifier loss summary")
plt.legend()
plt.grid()
plt.show()

print("---------------------------------------------------------------------------------------------------------------------------------------")
###########################################################################################################################################################





