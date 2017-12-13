import sys
import tensorflow as tf #need to run source activate tensorflow to load anaconda environment that uses tensorflow when running for the first time in a new terminalo
import random
import gym
import gym_pull
import numpy as np


numNeurons = 20
NUM_ACTIONS = 6
learningRate = 0.02
#LEARNING_RATE = 0.01



def createMarioNetwork():
##    startState = sum(row for row in startState, [])
##    endState = sum(row for row in endState, [])


    env = gym.make("ppaquette/SuperMarioBros-1-1-Tiles-v0")
    state = env.reset()
    state = np.reshape(state, -1)
     
    ##########################################################################################
    ############################# create input layer #########################################
    ##########################################################################################

    inputLayer = tf.placeholder(tf.float32, shape = [None, len(state)])

    ##########################################################################################
    ############################# create hidden layer #########################################
    ##########################################################################################
    W_hidden = tf.Variable(tf.truncated_normal([len(state), numNeurons], stddev = 0.1))

    b_hidden = tf.Variable(tf.constant(0.1, shape=[numNeurons]))


    net_hidden = tf.matmul(inputLayer, W_hidden) + b_hidden

    # apply the activation function to the vector containing the activations/Sj/net of the hidden layer
    out_hidden = tf.sigmoid(net_hidden) # this will get fed through the weights of the output layer

    ##########################################################################################
    ############################# create output layer #########################################
    ##########################################################################################
    W_output = tf.Variable(tf.truncated_normal([numNeurons, NUM_ACTIONS], stddev = 0.1))

    b_output = tf.Variable(tf.constant(0.1, shape=[NUM_ACTIONS]))

    # do the same matrix multiplication process to feed the output of the hidden layer through the weights of the
        # neurons in the output layer
    net_output = tf.matmul(out_hidden, W_output) + b_output

    # create the predictions of the neural network by applying activation function to the output ( a vector ) of the
        # output layer
    
    predictions = tf.nn.softmax(net_output)

    curMask = tf.placeholder(tf.float32, shape = [NUM_ACTIONS], name = "mask")
    #nextMask = tf.placeholder(tf.float32, shape = [1, NUM_ACTIONS], name = "mask")
    rewardTensor = tf.placeholder(tf.float32, shape = [NUM_ACTIONS], name = "reward")
    nextQ = tf.placeholder(tf.float32, shape = [NUM_ACTIONS], name = "nextQ")
    curQ = tf.placeholder(tf.float32, shape = [NUM_ACTIONS], name = "curQ")


    
    cost = tf.reduce_sum(((curMask * rewardTensor) + (curMask * nextQ) - (curMask * net_output))**2)
 #   cost = tf.reduce_sum(0.5 * (y - predictions) * (y - predictions))

    trainer = tf.train.AdamOptimizer(learningRate).minimize(cost)
    
    sess = tf.Session()
    # All variables needs to be initialize at the start of training otherwise they may hold
        # remnant values from previous execution.
    init = tf.initialize_all_variables().run(session=sess)

    experiences = []
    subset = []
    marioMove = [0, 0, 0, 0, 0, 0]



##    env = gym.make("ppaquette/SuperMarioBros-1-1-Tiles-v0")
##    state = env.reset()
    
    reward = 0

    while(True):
        observation, reward, done, info = env.step(env.action_space.sample())
        if(done):
            print("\n\nWE DEAD\n\n")
            env.reset()
##
##        print("state is: ", state)
##        print("type: ", type(state))

        
        state = np.reshape(state, (1, 208))
##        
##
##
##        print("state is now: ", state)
##        print("type: ", type(state))
##
##
##
##        print("reward is: ", reward)
       # state = sum([row for row in state], [])
        
        predictedMove = sess.run(predictions, feed_dict = {inputLayer : state})
        predictedMove = predictedMove[0]
        #print("predicted move is:", predictedMove)
        
        argmaxIndex = getMaxIndex(predictedMove)
        marioMove = [0] * 6
        marioMove[argmaxIndex] = 1 # TODO: NEED TO INCLUDE MAPPING OF COMBINATION MOVES
        
        nextState, nextReward, done, info = env.step(marioMove)
        
        
 #       nextState = sum([row for row in nextState], [])
        nextState = np.reshape(nextState, (1, 208))
        
        ex = (state, getZeroedOutArr(predictedMove), getReward(argmaxIndex, len(predictedMove), reward), nextState, argmaxIndex)
        experiences.append(ex)

        state = nextState
        
        #get the subset of experiences
        indices = set()
        #print("about to hit this while loop")
        while len(indices) < (int(len(experiences) * .25)):
            indices.add(random.randint(0, len(experiences)-1))
        #print("\n\n indices:", indices)
        #print("size of experiences:", len(experiences))
        
        subset = [experiences[i] for i in indices]
        


        #print("about to loop through experiences")
        for experience in subset:
            # extract s, st+1, action, and reward
            q = experience[0]
            r = experience[2]
            nQ = experience[3]
            #maxIndex = experience[4]
            #print("Got experience")
            
            
            nQ_QVal = sess.run(predictions, feed_dict = {inputLayer : nQ})
            nQ_QVal = nQ_QVal[0]
            #print("prediction for next move:", nQ_QVal)
            
            
 #           maskNQ = (nQ_QVal)
            q_QVal = sess.run(predictions, feed_dict = {inputLayer : q})
            
            q_QVal = q_QVal[0]
            maxIndex = getMaxIndex(q_QVal)
            
            
            nqMaxIndex = getMaxIndex(nQ_QVal)
            nQ_QVal[maxIndex] = nQ_QVal[nqMaxIndex]

            
            maskCurQ = getMask(len(q_QVal), maxIndex)
##            print("about to run trainer")
##            print("r is:", r)
##            print("maskCurQ: ", maskCurQ)
##            print("maskCurQ: ", nQ_QVal)
##
##            print("curMask:", curMask)
##            print("reward:", reward)
##            print("inputLayer:", q)
##            print("nextQ:", nextQ)

            
            
            sess.run(trainer, feed_dict = {curMask : maskCurQ, rewardTensor : r, inputLayer : q, nextQ : nQ_QVal}) #training
              


            

def getMask(length, index):
    mask = [0] * length
    mask[index] = 1
    return mask
            

def getReward(maxIndex, length, r):
    print("maxIndex is: ", maxIndex, "length is:", length, "r is:", r)
    reward = [0] * length
    
    reward[maxIndex] = r

    return reward
        
def getMaxIndex(arr):
    return list(arr).index(max(arr))

              

def getZeroedOutArr(arr):
    maxIndex = -1
    maxVal = -10000000000
    mask = [0] *  len(arr)
    maxIndex = getMaxIndex(arr)
    

    

    mask[maxIndex] = arr[maxIndex]

    return mask
    



##def neuralNetwork(train, test, numAttributes, numLabels, learningRate, numNeurons, numIter):
##
##    
##    # print("len test[0]: ", len(test[0]), ", len test[1]: ", len(test[1]))
##    # print("len train[0]: ", len(train[0]), ", len train[1]: ", len(train[1]))
##    #
##    # print("numblabels: ", numLabels)
##    # print("numAttributes: ", numAttributes, "\n\n\n\n\n")
##
##    # for i in range(0, len(train[0])):
##    #     print("testSet instance: ", instances[i])
##
##
##    ##########################################################################################
##    ############################# create input layer #########################################
##    ##########################################################################################
##
##    x = tf.placeholder(tf.float32, shape = [None, numAttributes])
##
##    ##########################################################################################
##    ############################# create hidden layer #########################################
##    ##########################################################################################
##    W_hidden = tf.Variable(tf.truncated_normal([numAttributes, numNeurons], stddev = 0.1))
##
##    b_hidden = tf.Variable(tf.constant(0.1, shape=[numNeurons]))
##    # multiply each incoming attribute value by its corresponding weight in W_hidden and add the bias weight
##        # do this with matrix multiplication:
##            #                        n1 n2 . . . nn <--- each matrix column represents
##            #                       _              _     neuron in hidden layer
##            #                      | w1 w1 . . . w1 |
##            #                      | w2 w2 . . . w2 |
##            # [x1, x2, ... , xd] * | .  .  . . . .  | = [ (instance_attributes dot n1_weights), (instance_attributes dot n1_weights), ... , (instance_attributes dot nd_weights)]
##            #          ^           | .  .  . . . .  |                                           ^
##            #          |           | .  .  . . . .  |                                           |
##            #    attributes for    | wd wd . . . wd |                                  gives us a vector representing the output
##            #    instance j         -              -                                   that each neuron gives us when we send
##            #    call this                  ^                                          the instance through it. Each index 'n' of this list
##            #    'x'.                       |                                                         d
##            #                       matrix that stores the weights                     is this value: Σ (wi*xi)   for neuron n.
##            #                       for every neuron in hidden layer.                                i=1
##            #                       Represents all of the neurons in our               We can then add this result to the weights of the bias terms
##            #                       hidden layer. This is the hidden layer.             to get the activation/net output/Sj for each neuron (since they
##            #                       Call this matrix 'W'.                               are the same dimension (1 x d)).
##            #
##            #
##            #   To get net/activation/Sj do x*W + b (where b is the weights for all of the bias terms. These bias term
##            #                                         weights are constant).
##    #
##    net_hidden = tf.matmul(x, W_hidden) + b_hidden
##
##    # apply the activation function to the vector containing the activations/Sj/net of the hidden layer
##    out_hidden = tf.sigmoid(net_hidden) # this will get fed through the weights of the output layer
##
##
##    ##########################################################################################
##    ############################# create output layer #########################################
##    ##########################################################################################
##    W_output = tf.Variable(tf.truncated_normal([numNeurons, numLabels], stddev = 0.1))
##
##    b_output = tf.Variable(tf.constant(0.1, shape=[numLabels]))
##
##    # do the same matrix multiplication process to feed the output of the hidden layer through the weights of the
##        # neurons in the output layer
##    net_output = tf.matmul(out_hidden, W_output) + b_output
##
##
##    # create the predictions of the neural network by applying activation function to the output ( a vector ) of the
##        # output layer
##    if numLabels == 1:
##        predictions = tf.sigmoid(net_output)
##    else:
##        predictions = tf.nn.softmax(net_output)
##
##
##    #create 98.139.180.181 true labels
##    y = tf.placeholder(tf.float32, shape = [None, numLabels])
##
##    # create training
##    if numLabels == 1:
##        cost = tf.reduce_sum(0.5 * (y - predictions) * (y - predictions))
##        print("cost is: ", cost)
##    else:
##        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=net_output))
##
##    trainer = tf.train.AdamOptimizer(learningRate).minimize(cost)
##
##
##    sess = tf.Session()
##    # All variables needs to be initialize at the start of training otherwise they may hold
##        # remnant values from previous execution.
##    init = tf.initialize_all_variables().run(session=sess)
##
##    # train the neural network
##    step = 0
##    #maxsteps = 100
##    maxsteps = numIter
##
##    #print("shape of predictions: ", predictions.get_shape())
##
##
##   # print("train[0]: ", train[0])
##
##    while step < maxsteps:
##        step += 1
##        #x was a place holder, now assign x to train[0] and y to train[1] with feed_dict
##            # if all you are doing is training only need to pass in trainer
##                # pass predict so that you can see what the output is
##                    # can pass in any other layer to see what the values for that layer are
##
##
##
##
##
##        _, p = sess.run([trainer, predictions], feed_dict = {x : train[0], y : train[1]}) # NEED TO FORMAT TRAINING SET AS [[attributes][labels]]
##        #print(p[:10])
##        # p = sess.run(predictions, feed_dict={x: test[0]})
##        # if step % 50 == 0:
##        #     confusionMatrix = getConfusionMatrix(p, train, numLabels)
##        #     print(calcAccuracy(confusionMatrix))
##
##
##    p = sess.run(predictions, feed_dict={x: test[0]})
##    #print("number of things I'm tryna predict: ", len(test[0]), "number of things I got:", len(p))
##   # print(predictions)
##
##
##    # for i in range(0, len(p)):
##    #     print("p[",i, "]: ", p[i])
##
##    return p
##
##
##
##def getParams():
##    if(len(sys.argv) < 7):
##        print("need to provide 6 input parameters:\n"
##              "1. path to file containing a data set\n"
##              "2. the number of neurons to use in the hidden layer\n"
##              "3. The learning rate to use during backpropogation\n"
##              "4. The number of iterations to use during training\n"
##              "5. The percentage of instances to use for a training set\n"
##              "6. A random seed as an integer")
##        sys.exit(0)
##
##
##
##    return sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
##
##
##def initializeTrainingAndTest():
##    testInstances = []
##    trainingInstances = []
##    testInstances.append([])
##    testInstances.append([])
##
##    trainingInstances.append([])
##    trainingInstances.append([])
##
##    return trainingInstances, testInstances
##
##
##def splitIntoTrainingAndTesting(instances, percentTraining, seed):
##
##    trainingInstances, testInstances = initializeTrainingAndTest()
##
##    random.seed(seed)
##    shuffled = list(instances)
##    random.shuffle(shuffled)
##    numInstances = int(len(shuffled))
##
##    numTraining = int(numInstances * percentTraining)
##
##
##    for i in range(0, len(shuffled)):
##        instance = shuffled[i]
##        if i < numTraining:
##            trainingInstances[0].append(instance.attributeValues)
##            trainingInstances[1].append(instance.labelValues)
##        else:
##            testInstances[0].append(instance.attributeValues)
##            testInstances[1].append(instance.labelValues)
##
##    return trainingInstances, testInstances
##
##
##
##def getMaxIndex(l):
##    max = -sys.maxsize
##    maxIndex = -sys.maxsize
##    for i in range(0, len(l)):
##        if l[i] > max:
##            max = l[i]
##            maxIndex = i
##
##    return maxIndex
##
##def getConfusionMatrix(predictions, testSet, numLabels):
##    actualLabels = testSet[1]
##    confusionMatrix = []
##
##    #print("numLabels is ", numLabels)
##
##    for i in range(0, numLabels):
##        confusionMatrix.append([])
##        for j in range(0, numLabels):
##            confusionMatrix[i].append(0)
##
##    for i in range(0, len(actualLabels)):
##       # print("actualLabels", i, ":", actualLabels[i])
##       # print("predictions", i, ":", predictions[i])
##       # print("max of prediction is:", max(predictions[i]))
##       # print("adding 1 to (", getMaxIndex(actualLabels[i]), ",", getMaxIndex(predictions[i]), ")")
##        confusionMatrix[getMaxIndex(actualLabels[i])][getMaxIndex(predictions[i])] += 1
##        #if testSet[i].index(max(testSet[i])) == predictions[i].index(max(predictions[i])):
##
##    #print(confusionMatrix)
##    return confusionMatrix
##
##
##def getConfusionMatrixBinary(predictions, testSet, numLabels, possibleAttributeVals, possibleLabelVals):
##    actualLabels = testSet[1]
##    confusionMatrix = []
##    for i in range(0, numLabels):
##        confusionMatrix.append([])
##        for j in range(0, numLabels):
##            confusionMatrix[i].append(0)
##
##
##
##    for i in range(0, len(actualLabels)):
##        predictionLabel = 0
##        print("predictions[", i, "]: ", predictions[i], "actualLabels[i]:", actualLabels[i])
##
##        if predictions[i] > 0.5:
##            predictionLabel = 1
##
##
##        confusionMatrix[actualLabels[i][0]][predictionLabel] += 1
##
##
##        #print("adding 1 to (", actualLabels[i][0], ",", predictionLabel, ")")
##
##
##       # print(predictions[i])
##
##    return confusionMatrix
##
##def printConfusionMatrix(confusionMatrix):
##    for i in range(0, len(confusionMatrix)):
##        for j in range(0, len(confusionMatrix[i])):
##            print(confusionMatrix[i][j], end=",")
##        print()
##
##def main():
##
##    filePath, numNeurons, learningRate, numIter, percentTraining, seed = getParams()
##    numNeurons = int(numNeurons)
##    learningRate = float(learningRate)
##    numIter = int(numIter)
##
##    instances, possibleLabels, possibleAttributeValues, nominal = parse_file.getInstances(filePath)
##
##    # for i in range(0, len(instances)):
##    #     print("testSet instance: ", instances[i])
##
##
##    if len(possibleLabels) > 2:
##        numLabels = len(possibleLabels)
##    else:
##        numLabels = 1
##
##
##
##
##    trainingSet, testSet = splitIntoTrainingAndTesting(instances, float(percentTraining), int(seed))
##
##
##    if nominal:
##        numAttributes = len(trainingSet[0][0])
##    else:
##        numAttributes = len(possibleAttributeValues)
##
##    predictions = neuralNetwork(trainingSet, testSet, numAttributes, numLabels, learningRate, numNeurons, numIter)
##
##
##    if numLabels == 1:
##        confusionMatrix = getConfusionMatrixBinary(predictions, testSet, len(possibleLabels), possibleAttributeValues, possibleLabels)
##    else:
##        confusionMatrix = getConfusionMatrix(predictions, testSet, len(possibleLabels))
##
##
##    print("possible lables:", possibleLabels)
##    printConfusionMatrix(confusionMatrix)
##
##    outputFilename = getOutputFileName(filePath, str(numNeurons), str(learningRate), str(numIter), str(percentTraining), str(seed))
##    outFile = open(outputFilename, "w")
##
##
##    for label in possibleLabels:
##        outFile.write(label.strip("\"")+",")
##    outFile.write("\n")
##    for i in range(0, len(confusionMatrix)):
##        for j in range(0, len(confusionMatrix[i])):
##            outFile.write(str(confusionMatrix[i][j]) + ",")
##        outFile.write(possibleLabels[i].strip("\"")+"\n")
##
##
##
##
##def getOutputFileName(inputFilename, numNeurons, learningRate, iterations, trainingPercent, seed):
##    inputFilenameParts = inputFilename.split(".")
##    inputFilename = inputFilenameParts[0]
##    #s_<DataSet>_<Neurons>n_<LearningRate>r_<Iterations>i_<TrainingPercentage>p_<Seed>.csv
##    rtn = "results_"+inputFilename+"_"+numNeurons+"n_"+learningRate+"r_"+iterations+"i_"+trainingPercent+"p_"+seed+".csv"
##    return rtn
##
##
##def calcAccuracy(confusionMatrix):
##    diagonalSum = 0.0
##    totalSum = 0.0
##
##    for i in range(0, len(confusionMatrix)):
##        for j in range(0, len(confusionMatrix[i])):
##            totalSum += confusionMatrix[i][j]
##            if i == j:
##                diagonalSum += confusionMatrix[i][j]
##
##
##    #print(confusionMatrix)
##    #print("diagonalSum:", diagonalSum, "totalSum:", totalSum)
##    return (diagonalSum / totalSum);



def main():
    createMarioNetwork()

main()

