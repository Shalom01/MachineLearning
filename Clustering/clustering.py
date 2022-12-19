'''
Created on Dec. 3, 2022

@author: Shalom
'''

import matplotlib.pyplot as plt
import numpy as np
import random as rand
import itertools

#############################FILE I/O##############################

twodpoints_file = open('twodpoints.txt', 'r')
threedpoints_file = open('threedpoints.txt', 'r')
seeds_file = open('seeds_dataset.txt', 'r')

inputs2D = []
inputs3D = []
seedsInputs = []

twodpoints_data = twodpoints_file.readlines()
threedpoints_data = threedpoints_file.readlines()
seeds_data = seeds_file.readlines()

for line in twodpoints_data:
    line = line.strip()
    point = line.split(",")
    point = [float(number) for number in point]
    inputs2D.append(point)
    
for line in threedpoints_data:
    line = line.strip()
    point = line.split(",")
    point = [float(number) for number in point]
    inputs3D.append(point)
    
for line in seeds_data:
    line = line.strip()
    point = line.split()
    point = [float(number) for number in point]
    point.pop()
    seedsInputs.append(point)
    
    
twodpoints_file.close()
threedpoints_file.close()

######################PART A: K-MEANS IMPLEMENTATION###############################

def kmeansCost(clusters): #function that calculates the kMeansCost of a cluster
    kmeans_cost = 0
    for cluster in clusters:
        if len(cluster) != 0:
            mean = [0 for d in range(len(cluster[0]))]
            for point in cluster: #add up all the points in the cluster
                mean = np.add(mean, point)
            mean = np.divide(mean, len(cluster)) #get the mean point
            
            for point in cluster:
                kmeans_cost += pow(np.linalg.norm(np.subtract(mean, point)), 2)
            
    return kmeans_cost

def Kmeans(inputs, means): ##the k-Means algorithm implementation
    #Calculate the clusters from the initial means
    clusters = []
    for mean in means:
        cluster = []
        for j in inputs:
            inCluster = True
            distance = np.linalg.norm(np.subtract(j, mean)) #find the distance to the mean
            for m in means:
                if np.linalg.norm(np.subtract(j, m)) < distance: #if there is another mean that has smaller distance
                    inCluster = False
                        
            if inCluster:
                cluster.append(j)
        clusters.append(cluster)
                     
    #update the centers (i.e., means)
    means = []
    for cluster in clusters:
        summation = [0 for i in range(len(inputs[0]))]
        for i in range(0, len(cluster)):
            summation = np.add(summation, cluster[i])
        mean = np.divide(summation, len(cluster))
        means.append(mean)
    
    previousCost = 0
    cost = kmeansCost(clusters) #retrieve the initial kmeansCost of the clusters

    while abs(previousCost - cost) >= 30: #while we have no converged
        previousCost = kmeansCost(clusters)
        clusters = []
        #compute the clusters
        for mean in means:
            cluster = []
            for j in inputs:
                inCluster = True
                distance = np.linalg.norm(np.subtract(j, mean)) #find the distance to the mean
                for m in means:
                    if np.linalg.norm(np.subtract(j, m)) < distance: #if there is another mean that has smaller distance
                        inCluster = False
                        
                if inCluster:
                    cluster.append(j)
            clusters.append(cluster)
                
        #update the centers (i.e., means)
        means = []
        for cluster in clusters:
            summation = [0 for i in range(len(inputs[0]))]
            for i in range(0, len(cluster)):
                summation = np.add(summation, cluster[i])
            mean = np.divide(summation, len(cluster))
            means.append(mean)
        
        cost = kmeansCost(clusters) #retrieve the kmeansCost of the clusters
        print(cost)
        
    #output formatting to that we output cluster list of size n containing values {1,..,k} indicating the cluster
    clusterAssignment = [1 for i in range(0, len(inputs))]    
    for i in range(0, len(inputs)): #for every input
        for j in range(0, len(clusters)): #check which cluster it is in
            for point in clusters[j]:
                if np.array_equal(point, inputs[i]):
                    clusterAssignment[i] = j + 1
    return clusterAssignment
     
###########################PART B: PLOTTING DATAPOINTS AND INITIALIZING MEANS BY HAND############################

##plot the 2D data
x1 = [column[0] for column in inputs2D]
x2 = [column[1] for column in inputs2D]

plt.plot(x1, x2,  'o', markersize=5, color = '#0000EE', alpha = 0.5)
plt.xlabel("X\u2081")
plt.ylabel("X\u2082")
plt.title("2-Dimensional Data Plot")
plt.show()

##Helper function for plotting 2D plots of clusters and their initial means
##Precondition: cluster data and mean data must be 2-dimensional, k cannot exceed 10 for proper visualization
def plot2DClusters(inputs, clusterAssignment, initial_means, title, grid):
    
    colors = ["#DC143C", "#3D9140", "#0000EE", "#8B8B00", "#B23AEE", "#EE1289", "#FF6103", "#8A360F", "#104E8B", "#8B7B8B"]
    colors = rand.sample(colors, len(initial_means))
    #plot the clusters
    for i in range(0, len(inputs)):
        color = colors[clusterAssignment[i] - 1]
        inputX1 = inputs[i][0]
        inputX2 = inputs[i][1]
        plt.plot(inputX1, inputX2, 'o', markersize=5, lw = 2, color=color, alpha = 0.6)
        
    #plot the initial means
    initial_meansX1 = [column[0] for column in initial_means] 
    intial_meansX2 = [column[1] for column in initial_means]
    plt.plot(initial_meansX1, intial_meansX2, 'o', markersize=3, lw = 2, color='black')

    plt.xlabel("X\u2081")
    plt.ylabel("X\u2082")
    if grid == True :
        plt.grid(True)
    plt.title(title)
    plt.show()

##INITIALIZATION METHOD 1: HAND SELECTION##
#INITIALIZATION 1#
manualMeans = []
manualMeans.append([-1.1811,12.79])
manualMeans.append([8.0021,7.6429])
manualMeans.append([1.0931,0.62184])

clusterAssignment = Kmeans(inputs2D, manualMeans)
plot2DClusters(inputs2D, clusterAssignment, manualMeans, "k-Means Output for 3 Manually Selected Means", False)

#INITIALIZATION 1#
manualMeans.clear()
manualMeans.append([1.138,0.28926])
manualMeans.append([1.9239,1.2669])
manualMeans.append([5.2221,9.316])

clusterAssignment = Kmeans(inputs2D, manualMeans)
plot2DClusters(inputs2D, clusterAssignment, manualMeans, "k-Means Output for 3 Manually Selected Means", False)

###########################PART C: METHOD 2 AND 3 OF INITIALIZING MEANS############################

##INITIALIZATION METHOD 2: RANDOM INTIALIZATION##
for i in range(0, 10): #10 trials of k-means algorithm with randomly (uniformly) selected 3 means
    randomMeans = []
    randomMeans.append(inputs2D[rand.randint(0, len(inputs2D) - 1)])
    randomMeans.append(inputs2D[rand.randint(0, len(inputs2D) - 1)])
    randomMeans.append(inputs2D[rand.randint(0, len(inputs2D) - 1)])
    
    clusterAssignment = Kmeans(inputs2D, randomMeans)
    plot2DClusters(inputs2D, clusterAssignment, randomMeans, "k-Means Output for 3 Randomly Selected Means (Trial #" + str(i + 1) + ")" , False)

##INITIALIZATION METHOD 3: DISTANCE INITIALIZATION##
def maximumDistanceMeans(inputs, k): #returns a list of k means initialized from the inputs according to method 3 of initialization. precondition: k >= 1
    means = []
    means.append(inputs[rand.randint(0, len(inputs) - 1)]) #select the first mean at random
    
    i = 1
    while i < k: #calculate k - 1 other means
        max_distance_sum = 0
        max_distance_point = means[0]
        
        for point in inputs:
            distance_sum = 0
            for mean in means:
                distance_sum += np.linalg.norm(np.subtract(mean, point))
            if distance_sum > max_distance_sum:
                max_distance_sum = distance_sum
                max_distance_point = point
        means.append(max_distance_point)
        i += 1

    return means

for i in range(0, 10): #10 trials of k-means algorithm with randomly (uniformly) selected 3 means
    method3Means = maximumDistanceMeans(inputs2D, 3)
    clusterAssignment = Kmeans(inputs2D, method3Means)
    plot2DClusters(inputs2D, clusterAssignment, method3Means, "k-Means Output for 3 Means Selected By Maximum Distance (Trial #" + str(i + 1) + ")"  , False)

###########################PART D: 2D-POINTS K-MEANS COST############################

#run 10 trials of k-Means algorithm on the 2D data points with k = 1, 2,...10 and calculate the cost
kTrials2DCost = []
kTrials = [k for k in range(1, 11)]

for k in kTrials:
    initial_means = maximumDistanceMeans(inputs2D, k) #initialize the means according to method 3
    clusterAssignment = Kmeans(inputs2D, initial_means) #calculate the k clusters
    
    ##create the clusters from the cluster assignment:
    clusters = []
    for i in range(0, len(initial_means)):
        cluster = []
        for j in range(0, len(inputs2D)):
            if clusterAssignment[j] - 1 == i:
                cluster.append(inputs2D[j])
        clusters.append(cluster)
        
    #calculate and record the cost for this k run
    kTrials2DCost.append(kmeansCost(clusters)) 

#plot cost per k plot
plt.plot(kTrials, kTrials2DCost, 'o-', markersize=5, color='#0000EE')
plt.xlabel("Cluster Number k")
plt.ylabel("k-Means Cost")
plt.title("k-Means Cost for Each k on the Two Dimensional Data")
plt.xticks(np.arange(1, 11, 1))
plt.show()

###########################PART E: 3D-POINTS K-MEANS COST############################

#Helper function for plotting 3D plots of clusters and their initial means. 
##Precondition: cluster data and mean data must be 3-dimensional, k cannot exceed 10 for proper visualization
def plot3DClusters(inputs, clusterAssignment, initial_means, title):
    colors = ["#DC143C", "#3D9140", "#0000EE", "#8B8B00", "#B23AEE", "#EE1289", "#FF6103", "#8A360F", "#104E8B", "#8B7B8B"]
    colors = rand.sample(colors, len(initial_means))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(0, len(inputs)): #plot the clusters
        color = colors[clusterAssignment[i] - 1] #get the cluster color
        inputX1 = inputs[i][0]
        inputX2 = inputs[i][1]
        inputX3 = inputs[i][2]
        ax.plot3D(inputX1, inputX2, inputX3, "o", markersize=8, lw = 2, color = color, alpha = 0.6)
        
    #plot the initial means
    initial_meansX1 = [column[0] for column in initial_means]
    intial_meansX2 = [column[1] for column in initial_means]
    intial_meansX3 = [column[2] for column in initial_means]
    ax.plot3D(initial_meansX1, intial_meansX2, intial_meansX3, "o", markersize=8, lw = 2, color = 'black')
    
    ax.set_xlabel("X\u2081", linespacing=3.0)
    ax.set_ylabel("X\u2082", linespacing=3.0)
    ax.set_zlabel("X\u2083", linespacing=3.0)

    plt.title(title)
    plt.show()

#run 10 trials of k-Means algorithm on the 3D data points with k = 1, 2,...10 and calculate the cost
kTrials3DCost = []

for k in kTrials:
    initial_means = maximumDistanceMeans(inputs3D, k) #initialize the means according to method 3
    clusterAssignment = Kmeans(inputs3D, initial_means) #calculate the k clusters
    
    ##create the clusters from the cluster assignment:
    clusters = []
    for i in range(0, len(initial_means)):
        cluster = []
        for j in range(0, len(inputs3D)):
            if clusterAssignment[j] - 1 == i:
                cluster.append(inputs3D[j])
        clusters.append(cluster)
    
    kTrials3DCost.append(kmeansCost(clusters))
    
#plot cost per k plot
plt.plot(kTrials, kTrials3DCost, 'o-', markersize=5, color='#0000EE')
plt.xlabel("Cluster Number k")
plt.ylabel("k-Means Cost")
plt.title("k-Means Cost for Each k on the Three Dimensional Data")
plt.xticks(np.arange(1, 11, 1))
plt.show()

#it appears that k = 4 is an ideal choice here, let's plot the output clusters on a 3D graph
initial_means = maximumDistanceMeans(inputs3D, 4)
clusterAssignment = Kmeans(inputs3D, initial_means)
plot3DClusters(inputs3D, clusterAssignment, initial_means, "k-Means Clustering for k=4 on 3-Dimensional Data")

###########################PART F: SEEDS DATA K-MEANS COST############################

kTrialsSeedsCost = []

for k in kTrials:
    initial_means = maximumDistanceMeans(seedsInputs, k) #initialize the means according to method 3
    clusterAssignment = Kmeans(seedsInputs, initial_means) #calculate the k clusters
    
    #create the clusters from the cluster assignment:
    clusters = []
    for i in range(0, len(initial_means)):
        cluster = []
        for j in range(0, len(seedsInputs)):
            if clusterAssignment[j] - 1 == i:
                cluster.append(seedsInputs[j])
        clusters.append(cluster)

    kTrialsSeedsCost.append(kmeansCost(clusters))

plt.plot(kTrials, kTrialsSeedsCost, 'o-', markersize=5, color='#0000EE')
plt.xlabel("Cluster Number k")
plt.ylabel("k-Means Cost")
plt.title("k-Means Cost for Every Trial on the Seeds Data")
plt.xticks(np.arange(1, 11, 1))
plt.show()

###########################PART G: SUBOPTIMAL CLUSTERINGS############################

dataset = [[2,1], [3,1], [3,2], [2,2], [-2, 1], [-3, 1], [-3,2], [-2, 2], [0, 8]] #chosen dataset

for data in dataset: #plot of the data
    plt.plot(data[0], data[1], 'o-', markersize=5, color='#0000EE')
axis = plt.gca()
plt.xlabel("X\u2081")
plt.ylabel("X\u2082")
plt.title("Dataset S")
plt.grid(True)
plt.show()

initial_means = maximumDistanceMeans(dataset, 2) #initialize the means according to method 3
clusterAssignment = Kmeans(dataset, initial_means) #calculate the k clusters
plot2DClusters(dataset, clusterAssignment, initial_means, "2-Means Clustering for S", True)

clusters = []
for i in range(0, len(initial_means)):
    cluster = []
    for j in range(0, len(dataset)):
        if clusterAssignment[j] - 1 == i:
            cluster.append(dataset[j])
    clusters.append(cluster)
    
cost_maxMeans = kmeansCost(clusters)
initial_means = list(itertools.combinations(dataset, 2))

cost_randomMeans = []
for means in initial_means:
    clusterAssignment = Kmeans(dataset, means) #calculate the k clusters
    
    clusters = []
    for i in range(0, len(initial_means)):
        cluster = []
        for j in range(0, len(dataset)):
            if clusterAssignment[j] - 1 == i:
                cluster.append(dataset[j])
        clusters.append(cluster)
    cost_randomMeans.append(kmeansCost(clusters))

trials = [i for i in range(1, 37)]
method3Costs = [cost_maxMeans for i in range(1, 37)]
plt.plot(trials, cost_randomMeans, markersize=2, color='blue', label = 'all possible initializations')
plt.plot(trials, method3Costs, markersize=2, color='red', label = 'initialized by third method')

plt.xticks(np.arange(0, 37, step=2))  # Set label locations.
plt.yticks(np.arange(40, 65, step=5))  # Set label locations.

plt.xlabel("Possible Means")
plt.ylabel("2-Means Cost")
plt.legend()
plt.grid(True)
plt.show()

initial_means = [[3,2],[-3,2]] #initialize the means according to method 3
clusterAssignment = Kmeans(dataset, initial_means) #calculate the k clusters
plot2DClusters(dataset, clusterAssignment, initial_means, "2-Means Clustering for S", True)

clusters = []
for i in range(0, len(initial_means)):
    cluster = []
    for j in range(0, len(dataset)):
        if clusterAssignment[j] - 1 == i:
            cluster.append(dataset[j])
    clusters.append(cluster)
print(kmeansCost(clusters))


#######################################################################################################




