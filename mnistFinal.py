# Experiment to test use of regularisation in Transfer Learning settings with MNIST digits, Graeme Blyth 2017

import tensorflow.examples.tutorials.mnist
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from utilities import weight_variable, bias_variable, conv2d, max_pool22, flatten_outer_dims, compute_fisher2, TaskBatch
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import scipy.linalg as slin
import os

# datasets
mnist = tensorflow.examples.tutorials.mnist.input_data.read_data_sets("MNIST_data/", one_hot=True)
mnist2 = tensorflow.examples.tutorials.mnist.input_data.read_data_sets("MNIST_data/", one_hot=False)

# model options
ewcMode = False
singleMode = False
searchMode = False
filePath = "/Users/graeme/private"
version = 'DropFull'

# model parameters
order = 2 #for norms
imageSize = 784
batchSize = 50
FbatchSize = 200
numIts = 20000
convFilters = 16
hidUnits = 32
etaInitial = 0.15
etaDecay = 0.998
imageDropout = 0.8 # keep probability
featureDropout = 0.5
masterLamdaF = 0.1
lamdaDef = 0.01
muDef = 0.01
zeta = 1e-10 # prevent log(0) and encourage PSD matrices
tasks = 10
zeroThresh = 1e-3
reps = 10


def saveEmbed(taskLoop):
    # embeddings for t-sne
    testImages = mnist.test.images[:5000]
    testLabels = mnist2.test.labels[:5000]

    # uncomment to select only embeddings for task
    # testImages, testLabels = TaskBatch(mnist.test.images,mnist.test.labels,task)
    # testLabels, dummy = TaskBatch(mnist2.test.labels,mnist.test.labels,task)

    #create embedding tensor
    embeddingIn = sess.run(hFC1, feed_dict={x:testImages, imageKeepProb: 1, featureKeepProb: 1})
    embeddingVar = tf.Variable(embeddingIn, trainable=False, name='embeddingVar')
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = 'embeddingVar'

    # Link this tensor to its metadata file (e.g. labels).
    np.savetxt('labels'+str(taskLoop)+'.tsv', testLabels, delimiter='\t')
    embedding.metadata_path = 'labels'+str(taskLoop)+'.tsv'

    #initialise and save tensor
    summary_writer = tf.summary.FileWriter(filePath)
    projector.visualize_embeddings(summary_writer, config)
    saver = tf.train.Saver()
    sess.run(tf.variables_initializer([embeddingVar]))
    saver.save(sess, os.path.join(filePath, 'mnist.ckpt' + str(taskLoop)))
    return 0


# define graph inputs
x = tf.placeholder(tf.float32, [None, imageSize])
ystar = tf.placeholder(tf.float32, [None, tasks])
yMask = tf.placeholder(tf.float32)
eta = tf.placeholder(tf.float32)
mu = tf.placeholder_with_default(tf.constant(muDef),[])
lamda = tf.placeholder_with_default(tf.constant(lamdaDef),[])
currTasks = tf.placeholder(tf.int32)
Wconv1star = tf.placeholder(tf.float32)
Wconv2star = tf.placeholder(tf.float32)
Wfc1star = tf.placeholder(tf.float32)
Wfc2star = tf.placeholder(tf.float32)
starVars = [Wconv1star,Wconv2star,Wfc1star,Wfc2star]
F0 = tf.placeholder(tf.float32)
F1 = tf.placeholder(tf.float32)
F2 = tf.placeholder(tf.float32)
F3 = tf.placeholder(tf.float32)
Fin = [F0,F1,F2,F3]
featureKeepProb = tf.placeholder(tf.float32)
imageKeepProb = tf.placeholder(tf.float32)
lamdaF = tf.placeholder(tf.float32)

# define graph variables
Wconv1 = weight_variable([5, 5, 1, convFilters],'Wconv1')
bConv1 = bias_variable([convFilters])
Wconv2 = weight_variable([3, 3, convFilters, convFilters],'Wconv2')
bConv2 = bias_variable([convFilters])
Wfc1 = weight_variable([7 * 7 * convFilters, hidUnits],'Wfc1')
bFC1 = bias_variable([hidUnits])
Wfc2 = weight_variable([hidUnits, tasks],'Wfc2')
bFC2 = bias_variable([tasks])

# convolution 1, compute 16 features from each 3x3 patch, then pool the 28x28 image into 14x14
# bias prevents dead neurons
xImage = tf.reshape(x, [-1,28,28,1])
xImage = tf.nn.dropout(xImage,imageKeepProb)
hConv1 = conv2d(xImage, Wconv1) #+ bConv1
# hConv1n = tf.contrib.layers.batch_norm(hConv1, center=True, scale=True)#, scope="bn1")
hConv1 = tf.nn.relu(hConv1)
hConv1n = tf.nn.dropout(hConv1,featureKeepProb)
hPool1 = max_pool22(hConv1n)

# convolution 2, compute 16 features from each 3x3 patch, then pool the 14x14 image into 7x7
hConv2 = conv2d(hPool1, Wconv2) #+ bConv2
# hConv2n = tf.contrib.layers.batch_norm(hConv2, center=True, scale=True)#, scope="bn2")
hConv2 = tf.nn.relu(hConv2)
hConv2n = tf.nn.dropout(hConv2,featureKeepProb)
hPool2 = max_pool22(hConv2n)

# non-linear layer, flatten processed image into vector; transform to 256 neurons and apply relu
hPool2flat = tf.reshape(hPool2, [-1,7*7*convFilters])
hFC1 = tf.matmul(hPool2flat, Wfc1)# + bFC1
# hFC1n = tf.contrib.layers.batch_norm(hFC1, center=True, scale=True)#, scope="bn1")
hFC1 = tf.nn.relu(hFC1)
hFC1n = tf.nn.dropout(hFC1,featureKeepProb)
yPre = tf.matmul(hFC1n, Wfc2)

# readout layer, softmax means elements of y form probability distribution (sum to one)
y = tf.nn.softmax(yPre + zeta)

# generate stats
hConv1sparsity = tf.nn.zero_fraction(hConv1)
hConv2sparsity = tf.nn.zero_fraction(hConv2)
hFC1sparsity = tf.nn.zero_fraction(hFC1)

varList = [Wconv1, Wconv2, Wfc1, Wfc2]


# define calculations
Fout = compute_fisher2(y,varList,batchSize,currTasks,zeta)

if ewcMode is True:
    ewcLoss = 0
    for var in range(len(varList)):
        ewcLoss += tf.reduce_sum(tf.matmul(flatten_outer_dims(tf.square(varList[var] - starVars[var])),Fin[var]))
else:
    ewcLoss = 0

#l-1 norm
if singleMode is True:
    currTasks = tf.cast(currTasks, dtype=tf.int32)
    # calculate l-1 norm from only current tasks
    l1 = tf.norm(Wfc2[:,currTasks[0]], ord=order) + tf.norm(Wfc2[:,currTasks[1]], ord=order) \
         + tf.norm(Wfc2[:,currTasks[2]], ord=order)
else:
    l1 = tf.norm(Wfc2, ord=1)

FreezeLoss = tf.reduce_sum(tf.square(Wfc2*yMask - Wfc2star*yMask))

crossEntropy = tf.reduce_mean(-tf.reduce_sum(ystar * tf.log(y + zeta), reduction_indices=[1])) + mu * FreezeLoss\
                #+ lamda * tf.nn.l2_loss(Wfc1) + mu * l1  # + lamdaF * ewcLoss
# crossEntropy = tf.reduce_mean(-tf.reduce_sum(ystar * tf.log(y + zeta), reduction_indices=[1]))# + lamda * tf.norm(Wfc1) + mu * l1

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(ystar, 1))
confusionMatrix = tf.contrib.metrics.confusion_matrix(tf.argmax(y,1), tf.argmax(ystar,1))

# function to run optimisation
trainStep = tf.train.GradientDescentOptimizer(eta).minimize(crossEntropy)

# correct_prediction gives list of booleans, take mean to measure % accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# useful for debugging
dydw = tf.gradients(crossEntropy, Wfc2)

# build graph
init = tf.global_variables_initializer()
sess = tf.Session()


def run(batchSize,numIts,etaIn,taskSets,taskLoop,starVarsIn,Fstar,yMaskIn,lamdaIn=[],muIn=[]):
    task = taskSets[:,taskLoop]
    accTrain = []
    accTest = []
    accTest2 = np.zeros([1000])
    activations = []
    weights = []
    accOrig = []
    testIndex = 0
    iterLoop = 0
    convTest = 2

    # for each iteration, randomly select a different batch of training & validation data from different sets
    while iterLoop < numIts and convTest > 1.01:
        batch_xs, batch_ys = mnist.train.next_batch(batchSize)
        if singleMode is True:
            batch_xs, batch_ys = TaskBatch(batch_xs, batch_ys, task)

        if ewcMode is True:
            batchAccuracy = sess.run(accuracy, feed_dict={x: batch_xs, ystar: batch_ys,
                                           Wconv1star: starVarsIn[0], Wconv2star: [1], Wfc1star: starVarsIn[2],
                                           Wfc2star: starVarsIn[3], lamdaF: starVarsIn[4], F0: Fstar[0], F1: Fstar[1],
                                           F2: Fstar[2], F3: Fstar[3], eta: etaIn, currTasks: taskSets[:,taskLoop],
                                            yMask: yMaskIn, imageKeepProb: 1, featureKeepProb: 1})
        else:
            batchAccuracy = sess.run(accuracy, feed_dict={x: batch_xs, ystar: batch_ys, eta: etaIn,
                                                          currTasks: taskSets[:,taskLoop], yMask:yMaskIn,
                                                          imageKeepProb: 1, featureKeepProb: 1})
        accTrain.append(batchAccuracy)

        if singleMode is True and iterLoop % 100 == 0 and taskLoop > 0:
            originalAccuracy = sess.run(accuracy, feed_dict={x: Oldbatch_xs, ystar: Oldbatch_ys,
                                                            yMask:np.ones(tasks), imageKeepProb: 1, featureKeepProb: 1})
            print("Original tasks accuracy %g" % (originalAccuracy))
            accOrig.append(originalAccuracy)

        # if iterLoop%5 == 0:
        val_xs, val_ys = mnist.validation.next_batch(batchSize*10)
        if singleMode is True:
            val_xs, val_ys = TaskBatch(val_xs, val_ys, task)
        accTestCurr, Wfc1out, Wfc2out, Wconv1out, Wconv2out, hC1sparOut, hC2sparOut, hFC1sparOut \
            = sess.run([accuracy, Wfc1, Wfc2, Wconv1, Wconv2, hConv1sparsity, hConv2sparsity, hFC1sparsity],
                       feed_dict={x: val_xs, ystar: val_ys, yMask:yMaskIn, imageKeepProb: 1, featureKeepProb: 1})

        # calculate weight sparsity statistics
        Wconv1out[Wconv1out < zeroThresh] = 0
        Wconv2out[Wconv2out < zeroThresh] = 0
        Wfc1out[Wfc1out < zeroThresh] = 0
        Wfc2out[Wfc2out < zeroThresh] = 0
        Wconv1spar = 1 - (np.count_nonzero(Wconv1out) / Wconv1out.size)
        Wconv2spar = 1 - (np.count_nonzero(Wconv2out) / Wconv2out.size)
        W1spar = 1 - (np.count_nonzero(Wfc1out)/Wfc1out.size)
        W2spar = 1 - (np.count_nonzero(Wfc2out)/Wfc2out.size)

        # save stats for output
        accTest2[testIndex] = accTestCurr
        activations.append([hC1sparOut+ zeta, hC2sparOut+ zeta, hFC1sparOut+zeta])
        weights.append([Wconv1spar, Wconv2spar, W1spar, W2spar])
        accTest.append(accTestCurr)
        testIndex += 1

        #convergence test
        if iterLoop > 500:
            convTest = np.mean(accTest[-20:]) / np.mean(accTest[-100:])
            # early stopping
            if accTest[-1] < accTest[-2] < accTest[-3] < accTest[-4] < accTest[-5]:
                convTest = 0

        if iterLoop%100 == 0:
            print("step %d, training accuracy %g, test accuracy %g." %
                  (iterLoop, batchAccuracy, accTestCurr))
            print("activation sparsity: hConv1 %g, hConv2 %g, hFC1 %g" % (hC1sparOut, hC2sparOut,hFC1sparOut))
            print("weight sparsity: conv1 %g, conv2 %g, Wfc1 %g, Wfc2 %g" % (Wconv1spar, Wconv2spar, W1spar, W2spar))
            print(convTest)

        if ewcMode is True:
            sess.run(trainStep, feed_dict={x: batch_xs, ystar: batch_ys, Wconv1star: starVarsIn[0], Wconv2star: [1],
                                           Wfc1star: starVarsIn[2], Wfc2star: starVarsIn[3], lamdaF: starVarsIn[4],
                                           F0: Fstar[0], F1: Fstar[1], F2: Fstar[2], F3: Fstar[3],eta: etaIn,
                                           currTasks: taskSets[:,taskLoop], yMask:yMaskIn,
                                           imageKeepProb: imageDropout, featureKeepProb: featureDropout})
        else:
            sess.run(trainStep, feed_dict={x: batch_xs, ystar: batch_ys,eta: etaIn, currTasks: taskSets[:,taskLoop],
                                           yMask:yMaskIn, Wfc2star: starVarsIn[3], lamdaF: starVarsIn[4],
                                           imageKeepProb: imageDropout, featureKeepProb: featureDropout})

        etaIn *= etaDecay
        iterLoop += 1

    # final statistics
    if singleMode is True:
        test_xs, test_ys = TaskBatch(mnist.test.images, mnist.test.labels, task)
        finalAccuracy = sess.run(accuracy,feed_dict={x: test_xs, ystar: test_ys, yMask:yMaskIn, imageKeepProb: 1, featureKeepProb: 1})
        fullSetAccuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, ystar: mnist.test.labels, yMask:yMaskIn,
                                                        imageKeepProb: 1, featureKeepProb: 1})
        print("Full Set Accuracy: %g" % (fullSetAccuracy))
        # print(sess.run(confusionMatrix, feed_dict={x: test_xs, ystar: test_ys}))
    else:
        finalAccuracy = sess.run(accuracy,feed_dict={x: mnist.test.images, ystar: mnist.test.labels, yMask:yMaskIn,
                                                     imageKeepProb: 1, featureKeepProb: 1})
    print("Rep %g, Training Rate %g, Steps %d, Final Training Accuracy %g, Final Test Accuracy %g" %
          (repLoop, etaIn, iterLoop, batchAccuracy, finalAccuracy))

    # calculate Fisher Information
    FisherMatrices = sess.run(Fout, feed_dict={x:batch_xs, currTasks: taskSets[:,taskLoop], yMask:yMaskIn,
                                               imageKeepProb: 1, featureKeepProb: 1})
    for v in range(len(FisherMatrices)):
        FisherMatrices[v] = slin.sqrtm(FisherMatrices[v])
    # sess.close()
    return finalAccuracy, iterLoop, accTest2, activations, weights, FisherMatrices

# loop over runs of the experiment
if singleMode is True:
    accuracies = np.zeros([reps,3])
    curves = np.zeros([reps,1000,3])
    activationsAll = np.zeros([reps,3000,3])
    weightsAll = np.zeros([reps,3000,4])

    steps = np.zeros([reps,3])
    frechet = np.zeros([reps,9])
    euclid = np.zeros([reps,9])
    batchSize = 300

    for repLoop in range(reps):
        numberIts = numIts
        etaIn = etaInitial
        lamdaFin = 0 # stop ewc regularsier for first iteration

        # create random sets of 3 out of 120 possible combinations
        taskSets = np.random.permutation(10)
        # taskSets = np.arange(0,9)
        taskSets = taskSets[:9].reshape((3,3))
        FisherMatrices = []
        weightMatrices = []

        sess.run(init)
        for taskLoop in range(3):
            yMaskIn = np.ones(tasks)
            for zeroLoop in range(3):
                yMaskIn[taskSets[zeroLoop,taskLoop]] = 0

            with sess.as_default():
                Wconv1starIn = Wconv1.eval()
                Wconv2starIn = Wconv2.eval()
                Wfc1starIn = Wfc1.eval()
                Wfc2starIn = Wfc2.eval()
            Oldbatch_xs, Oldbatch_ys = mnist.train.next_batch(FbatchSize * 10)
            if taskLoop == 1:
                oldTask = taskSets[:, taskLoop - 1]
                Oldbatch_xsC, Oldbatch_ysC = TaskBatch(Oldbatch_xs, Oldbatch_ys, oldTask)
                oldTasksIn = oldTask
            if taskLoop == 2:
                oldTask1 = taskSets[:, taskLoop - 1]
                oldTask2 = taskSets[:, taskLoop - 2]
                Oldbatch_xs1, Oldbatch_ys1 = TaskBatch(Oldbatch_xs, Oldbatch_ys, oldTask1)
                Oldbatch_xs2, Oldbatch_ys2 = TaskBatch(Oldbatch_xs, Oldbatch_ys, oldTask2)
                Oldbatch_xsC = np.concatenate((Oldbatch_xs1, Oldbatch_xs2), axis=0)
                Oldbatch_ysC = np.concatenate((Oldbatch_ys1, Oldbatch_ys2), axis=0)
                oldTasksIn = np.concatenate((oldTask1,oldTask2),axis=0)
            if taskLoop > 0:
                OldBatch = np.concatenate((Oldbatch_xsC,Oldbatch_ysC), axis = 1)
                batchIndex = np.random.permutation(len(OldBatch))
                Oldbatch_xs = OldBatch[batchIndex[:FbatchSize], :imageSize]
                Oldbatch_ys = OldBatch[batchIndex[:FbatchSize], imageSize:]
                Fstar = sess.run(Fout,feed_dict={x: Oldbatch_xs, currTasks: oldTasksIn, yMask:yMaskIn,
                                                 imageKeepProb: 1, featureKeepProb: 1})
            else:
                Fstar = sess.run(Fout,feed_dict={x: Oldbatch_xs, currTasks: [0,1,2], yMask:yMaskIn,
                                                 imageKeepProb: 1, featureKeepProb: 1})
            starVarsIn = [Wconv1starIn, Wconv2starIn, Wfc1starIn, Wfc2starIn, lamdaFin]


            finalAccuracy, iterLoop, curve, activations, weights, FisherOut = \
                run(batchSize, numberIts, etaIn, taskSets, taskLoop, starVarsIn, Fstar, yMaskIn)
            accuracies[repLoop,taskLoop] = finalAccuracy
            if taskLoop > 0:
                taskActivations = np.append(taskActivations, activations, axis=0)
                taskWeights = np.append(taskWeights, weights, axis=0)
            else:
                taskActivations = activations
                taskWeights = weights
            steps[repLoop,taskLoop] = iterLoop
            curves[repLoop,:,taskLoop] = curve
            FisherMatrices.append(FisherOut)
            weightMatrices.append(sess.run([Wconv1,Wconv2,Wfc1]))
            # etaIn *=  etaDecay ** iterLoop
            lamdaFin = masterLamdaF

            # save embeddings, weight matrices for task
            saveEmbed(taskLoop)
            Wfc2out, Wconv1out = sess.run([Wfc2, Wconv1])
            np.savetxt("wfc2" + version + str(taskLoop) + ".csv", Wfc2out, delimiter=',')
            np.save("wconv1" + version + str(taskLoop), Wconv1out)

        activationsAll[repLoop,:len(taskActivations),:] = taskActivations
        weightsAll[repLoop,:len(taskWeights),:] = taskWeights
        W1, W2, W3 = weightMatrices

        # calculate Frechet and Euclidean distances
        F1, F2, F3 = FisherMatrices
        for v in range(len(varList) - 1):
            frechet[repLoop, 0 + v*3] = 0.5 * np.linalg.norm(F1[v] - F2[v])
            frechet[repLoop, 1 + v*3] = 0.5 * np.linalg.norm(F1[v] - F3[v])
            frechet[repLoop, 2 + v*3] = 0.5 * np.linalg.norm(F2[v] - F3[v])
            euclid[repLoop, 0 + v*3] = 0.5 * np.linalg.norm(W1[v] - W2[v])
            euclid[repLoop, 1 + v*3] = 0.5 * np.linalg.norm(W1[v] - W3[v])
            euclid[repLoop, 2 + v*3] = 0.5 * np.linalg.norm(W2[v] - W3[v])

    accuracies = np.mean(accuracies,axis=0)
    steps = np.mean(steps,axis=0)

    activationsAll[activationsAll==0] = np.nan
    activationsMean = np.nanmean(activationsAll,axis=0)
    activationsStd = np.nanstd(activationsAll,axis=0)
    weightsAll[weightsAll==0] = np.nan
    weightsMean = np.nanmean(weightsAll,axis=0)
    weightsStd = np.nanstd(weightsAll,axis=0)

    curves[curves==0] = np.nan
    curvesMean = np.nanmean(curves,axis=0)
    std = np.nanstd(curves, axis=0)

    frechetMean = np.mean(frechet,axis=0)
    frechetStd = np.std(frechet,axis=0)
    euclidMean = np.mean(euclid,axis=0)
    euclidStd = np.std(euclid, axis=0)
    for v in range(len(varList) - 1):
        print('%s Frechet Distances:' %(varList[v].name))
        print('Task1-Task2: %g' % (frechetMean[0+v*3]))
        print('Task1-Task3: %g' % (frechetMean[1+v*3]))
        print('Task2-Task3: %g' % (frechetMean[2+v*3]))

        print('%s Euclidean Distances:' %(varList[v].name))
        print('Task1-Task2: %g' % (euclidMean[0+v*3]))
        print('Task1-Task3: %g' % (euclidMean[1+v*3]))
        print('Task2-Task3: %g' % (euclidMean[2+v*3]))

    np.savetxt("frechet" + version + ".csv",np.stack((frechetMean,frechetStd),axis=-1), delimiter=',')
    np.savetxt("euclid" + version + ".csv",np.stack((euclidMean,euclidStd),axis=-1), delimiter=',')

    np.savetxt("curves" + version + ".csv",np.concatenate((curvesMean,std),axis=1),
               delimiter=',', header='task1,task2,task3,std1,std2,std3')
    np.savetxt("activationSpar" + version + ".csv",np.concatenate((activationsMean,activationsStd),axis=1),
                                                                  delimiter=',', header='hconv1,hconv2,hfc1,std1,std2,std3')
    np.savetxt("weightSpar" + version + ".csv",np.concatenate((weightsMean,weightsStd),axis=1),
                                                              delimiter=',', header='wconv1,wconv2,wfc1,wfc2,std1,std2,std3,std4')


elif searchMode is True:
    validationLoops = 10
    lamLow, lamHigh = [0,1]
    muLow, muHigh = [0,1]
    meanValLosses = np.zeros([validationLoops])
    stdValLosses = np.zeros([validationLoops])
    parameters = np.zeros([validationLoops,2])

    # loop to validate best parameters
    for valLoops in range(validationLoops):
        valLosses = []
        lamdaCurr = np.random.uniform(lamLow,lamHigh)
        muCurr = np.random.uniform(muLow,muHigh)
        print("lamda %g, mu %g" % (lamdaCurr,muCurr))
        for testLoops in range(3):
            valLosses.append(run(batchSize,numIts,etaInitial,np.zeros((3,3)),0,0,0,lamdaCurr,muCurr))
        meanValLosses[valLoops] = np.mean(valLosses)
        stdValLosses[valLoops] = np.std(valLosses)
        parameters[valLoops,:] = lamdaCurr, muCurr
        valLoops +=1

    # save
    print(parameters)
    print(meanValLosses)
    print(stdValLosses)
    np.savetxt("ValLosses.csv", np.concatenate((parameters,np.stack((meanValLosses, stdValLosses), axis=-1)), axis=1),
               delimiter=',')
    starParamIndex = np.argmax(meanValLosses)
    #print("lamdaStar %g, muStar %g" % (parameters[0,starParamIndex],parameters[1,starParamIndex]))
    fig = plt.figure(5)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(parameters[0, :], parameters[1, :], meanValLosses)
    plt.show()

else:
    # loop to validate best learning rate(eta)
    # etaIndex = 0
    # eta = etaInitial
    # meanAccuracy = np.zeros([7])
    # while eta < 0.5:
    #     accuracy = []
    #     for testLoops in range(3):
    #         accuracy.append(run(batchSize,numIts,eta,False))
    #     meanAccuracy[etaIndex] = np.mean(accuracy)
    #     eta += 0.1
    #     etaIndex +=1
    # print(meanAccuracy)

    etaIn = 0.15  #np.argmax(meanAccuracy)*0.1 + 0.05
    repLoop = 1
    taskLoop = 1
    sess.run(init)
    void = run(batchSize,numIts,etaIn,np.zeros((3,3)),0,np.zeros([5]),0,np.ones(tasks))
    saveEmbed(taskLoop)
    Wfc2out, Wconv1out = sess.run([Wfc2, Wconv1])
    np.savetxt("wfc2" + version + str(taskLoop) + ".csv", Wfc2out, delimiter=',')
    np.save("wconv1" + version + str(taskLoop), Wconv1out)




