# Experiment to test use of regularisation in Transfer Learning settings with modified
# MNIST digits and unrelated tasks, Graeme Blyth 2017

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from utilities import weight_variable, bias_variable, conv2d, max_pool22, compute_fisher, TaskBatch2
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import scipy.io as sio
import scipy.linalg as slin
import pandas as pd
import os

# load data and create pandas dataframes
mnist2All = sio.loadmat('mnistTestNew.mat')
mnist2All = pd.DataFrame(mnist2All['allData'])
mnist2train = mnist2All.iloc[:55000,:784+16]
mnist2val = mnist2All.iloc[55000:60000,:784+16]
mnist2test = mnist2All.iloc[60000:,:784+16]

domAdap = False
version = 'l1'
order = 1
filePath = "/Users/graeme/private3"

hidUnits = 32
convFilters = 16
tasks = 16
imageDropout = 1 #keep probability
featureDropout = 1


def model(switchIn, hidUnits):
    # model parameters
    imageSize = 784
    batchSize = 50
    gamma = 0.9  # momentum
    lamda = 0.01
    mu = 0.01
    zeta = 1e-10  # to prevent log(0) and encourage PSD matrices

    numIts = 10000
    etaInitial = 0.15
    etaDecay = 0.998
    zeroThresh = 1e-2
    reps = 10

    def saveEmbed():
        # embeddings for t-sne
        testImages = mnist2All.iloc[60000:65000, :imageSize]
        testLabels = mnist2All.iloc[60000:65000, imageSize+16:]

        # create embedding tensor
        embeddingIn = sess.run(hFC1, feed_dict={x: testImages, imageKeepProb: 1, featureKeepProb: 1})
        embeddingVar = tf.Variable(embeddingIn, trainable=False, name='embeddingVar')
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = 'embeddingVar'

        # Link this tensor to its metadata file (e.g. labels).
        testLabels.to_csv('labelsOrtho.tsv', sep='\t', header=['digit','stroke','noise'], index=False)
        embedding.metadata_path = 'labelsOrtho.tsv'

        # initialise and save tensor
        summary_writer = tf.summary.FileWriter(filePath)
        projector.visualize_embeddings(summary_writer, config)
        saver = tf.train.Saver()
        sess.run(tf.variables_initializer([embeddingVar]))
        saver.save(sess, os.path.join(filePath, 'mnist.ckpt'))
        summary_writer.close()
        return 0

    # define graph inputs
    x = tf.placeholder(tf.float32, [None, 784])
    ystar = tf.placeholder(tf.float32, [None, tasks])
    yMask = tf.placeholder(tf.float32, [10])
    eta = tf.placeholder(tf.float32)
    switch = tf.placeholder(tf.float32, [3])
    featureKeepProb = tf.placeholder(tf.float32)
    imageKeepProb = tf.placeholder(tf.float32)
    Wfc2star = tf.placeholder(tf.float32)

    # convolution 1, compute 16 features from each 3x3 patch, then pool the 28x28 image into 14x14
    # bias prevents dead neurons
    Wconv1 = weight_variable([5, 5, 1, convFilters],'Wconv1')
    bConv1 = bias_variable([convFilters])
    xImage = tf.reshape(x, [-1,28,28,1])
    xImage = tf.nn.dropout(xImage,imageKeepProb)
    hConv1 = conv2d(xImage, Wconv1) #+ bConv1
    hConv1n = tf.contrib.layers.batch_norm(hConv1, center=True, scale=True)
    hConv1n = tf.nn.relu(hConv1n)
    hConv1n = tf.nn.dropout(hConv1n,featureKeepProb)
    hPool1 = max_pool22(hConv1n)

    # skip connection
    # hSkipFlat = tf.reshape(hPool1[:,:,:,0], [-1, 14 * 14 * 1])
    # Wskip = weight_variable([14 * 14 * convFilters, hidUnits],'Wskip')

    # convolution 2, compute 16 features from each 3x3 patch, then pool the 14x14 image into 7x7
    Wconv2 = weight_variable([3, 3, convFilters, convFilters],'Wconv2')
    bConv2 = bias_variable([convFilters])
    hConv2 = conv2d(hPool1, Wconv2) #+ bConv2
    hConv2n = tf.contrib.layers.batch_norm(hConv2, center=True, scale=True)
    hConv2n = tf.nn.relu(hConv2n)
    hConv2n = tf.nn.dropout(hConv2n,featureKeepProb)
    hPool2 = max_pool22(hConv2n)

    # non-linear layer, flatten processed image into vector; transform to 256 neurons and apply relu
    Wfc1 = weight_variable([7 * 7 * convFilters, hidUnits],'Wfc1')
    bFC1 = bias_variable([hidUnits])
    hPool2flat = tf.reshape(hPool2, [-1,7*7*convFilters])
    hFC1 = tf.matmul(hPool2flat, Wfc1) #+ bFC1 #+ hSkipFlat #tf.matmul(hSkipFlat, Wskip)
    hFC1n = tf.contrib.layers.batch_norm(hFC1, center=True, scale=True)
    hFC1n = tf.nn.relu(hFC1n)
    hFC1n = tf.nn.dropout(hFC1n,featureKeepProb)

    # zero labels for unused tasks
    ymaskFull = tf.concat([switch[0]*yMask,tf.tile([switch[1]],[3]),tf.tile([switch[2]],[3])],0)
    OldMaskFull = tf.concat([switch[0]*(tf.ones([10])-yMask),tf.zeros([6])],0)

    ystar *= ymaskFull

    # readout layer, softmax per classification task
    Wfc2 = weight_variable([hidUnits, tasks],'Wfc2')
    bFC2 = bias_variable([tasks])
    yRaw = tf.matmul(hFC1n, Wfc2)# + bFC2
    p1 = tf.nn.softmax(yRaw[:,:10]+ zeta) * switch[0]
    p2 = tf.nn.softmax(yRaw[:,10:13]+ zeta) * switch[1]
    p3 = tf.nn.softmax(yRaw[:,13:]+ zeta) * switch[2]
    y = tf.concat([p1,p2,p3],1)

    # generate stats
    hConv1sparsity = tf.nn.zero_fraction(hConv1n)
    hConv2sparsity = tf.nn.zero_fraction(hConv2n)
    hFC1sparsity = tf.nn.zero_fraction(hFC1n)

    # calculations
    Wfc2m = Wfc2 * ymaskFull

    #l_2,1 norm
    l21 = 0
    for l21loop in range(hidUnits):
        l21 += tf.norm(Wfc2m[l21loop,:])

    # orthogonality penalty, using masking to freeze inactive digits
    orthoPen = tf.norm(tf.matmul(tf.transpose(Wfc2m[:,:10]),Wfc2m[:,10:13]))**2\
                + tf.norm(tf.matmul(tf.transpose(Wfc2m[:,:10]),Wfc2m[:,13:]))**2\
                + tf.norm(tf.matmul(tf.transpose(Wfc2m[:,13:]),Wfc2m[:,10:13]))**2

    currTasks = tf.where(tf.not_equal(yMask,tf.constant(0, dtype=tf.float32)))
    currTasks = tf.cast(currTasks,dtype=tf.int32)
    l1 = tf.norm(Wfc2[:,currTasks[0,0]], ord=order) + tf.norm(Wfc2[:,currTasks[1,0]], ord=order) + tf.norm(Wfc2[:,currTasks[2,0]], ord=order)\
        + tf.norm(Wfc2[:,10:], ord=order)

    FreezeLoss = tf.reduce_sum(tf.square((Wfc2 - Wfc2star) * OldMaskFull))

    crossEntropy1 = tf.reduce_mean(-tf.reduce_sum((ystar+zeta) * tf.log(y + zeta), reduction_indices=[1]))\
                   + lamda * l1 + mu* FreezeLoss
    crossEntropy2 = tf.reduce_mean(-tf.reduce_sum((ystar+zeta) * tf.log(y + zeta), reduction_indices=[1])) + mu* FreezeLoss\
                    + mu*l21 + lamda*orthoPen
    crossEntropy3 = tf.reduce_mean(-tf.reduce_sum((ystar+zeta) * tf.log(y + zeta), reduction_indices=[1])) + mu* FreezeLoss

    # crossEntropy = (XentVersion[0]+XentVersion[1]) * crossEntropy1 + XentVersion[2] * crossEntropy2 + XentVersion[3] * crossEntropy3
    crossEntropy = crossEntropy1

    correct_prediction1 = tf.equal(tf.argmax(p1, 1), tf.argmax(ystar[:,:10], 1))
    correct_prediction2 = tf.equal(tf.argmax(p2, 1), tf.argmax(ystar[:,10:13], 1))
    correct_prediction3 = tf.equal(tf.argmax(p3, 1), tf.argmax(ystar[:,13:], 1))

    #confusionMatrix = tf.contrib.metrics.confusion_matrix(tf.argmax(y,1), tf.argmax(ystar,1))

    # function to run optimisation
    # trainStep = tf.train.GradientDescentOptimizer(eta).minimize(crossEntropy)
    trainStep = tf.train.MomentumOptimizer(eta,gamma,use_nesterov=True).minimize(crossEntropy)

    # correct_prediction gives list of booleans, take mean to measure % accuracy
    accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32)) * switch[0]
    accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32)) * switch[1]
    accuracy3 = tf.reduce_mean(tf.cast(correct_prediction3, tf.float32)) * switch[2]
    accuracy = (accuracy1 + accuracy2 + accuracy3) / tf.reduce_sum(switch)

    # single task loss
    loss1 = tf.reduce_mean(-tf.reduce_sum(ystar[:,:10] * tf.log(p1 + zeta), reduction_indices=[1]))
    loss2 = tf.reduce_mean(-tf.reduce_sum(ystar[:,10:13] * tf.log(p2 + zeta), reduction_indices=[1]))
    loss3 = tf.reduce_mean(-tf.reduce_sum(ystar[:,13:] * tf.log(p3 + zeta), reduction_indices=[1]))

    varList = [Wconv1, Wconv2, Wfc1, Wfc2]

    dydw = tf.gradients(crossEntropy, Wfc2)

    Fout = compute_fisher(y,varList,batchSize,zeta)

    # build graph
    sess = tf.Session()
    init = tf.global_variables_initializer()


    def run(batchSize,numIts,etaIn,switchIn, taskSets, taskLoop, yMaskIn):
        TaskMask = np.concatenate([np.tile([switchIn[0]], [10]), np.tile([switchIn[1]], [3]), np.tile([switchIn[2]], [3])], 0)
        task = taskSets[:, taskLoop]
        accTrain = []
        accTest = []
        accTest2 = np.zeros([1000])
        accDigit = np.zeros([1000])
        accStroke = np.zeros([1000])
        accNoise = np.zeros([1000])
        activations = []
        weights = []
        testIndex = 0
        iterLoop = 0
        convTest = 2

        # for each iteration, randomly select a different batch of training & validation data from different sets
        while iterLoop < numIts and convTest > 1.01:
            batch = mnist2train.sample(batchSize*2)
            if domAdap is True :
                batch = TaskBatch2(batch, imageSize, task)
            batch_xs = batch.iloc[:,:imageSize]
            batch_ys = batch.iloc[:,imageSize:]
            batch_ys *= TaskMask

            sess.run(trainStep, feed_dict={x: batch_xs, ystar: batch_ys, eta: etaIn, switch:switchIn,
                                           imageKeepProb: imageDropout, featureKeepProb: featureDropout, yMask:yMaskIn,
                                           Wfc2star: Wfc2starIn})
            etaIn *= etaDecay

            batchAccuracy = sess.run(accuracy, feed_dict={x: batch_xs, ystar: batch_ys, switch:switchIn,
                                     imageKeepProb: 1, featureKeepProb: 1, yMask:yMaskIn, Wfc2star: Wfc2starIn})

            accTrain.append(batchAccuracy)

            # generate weight sparsity statistics
            Wfc1out, Wfc2out, Wconv1out, Wconv2out = sess.run([Wfc1, Wfc2, Wconv1, Wconv2])
            Wconv1out[Wconv1out < zeroThresh] = 0
            Wconv2out[Wconv2out < zeroThresh] = 0
            Wfc1out[Wfc1out < zeroThresh] = 0
            Wfc2out[Wfc2out < zeroThresh] = 0
            Wconv1spar = 1 - (np.count_nonzero(Wconv1out) / Wconv1out.size)
            Wconv2spar = 1 - (np.count_nonzero(Wconv2out) / Wconv2out.size)
            W1spar = 1 - (np.count_nonzero(Wfc1out) / Wfc1out.size)
            W2spar = 1 - (np.count_nonzero(Wfc2out) / Wfc2out.size)

            # if iterLoop%5 == 0:
            val = mnist2val.sample(batchSize*10)
            if domAdap is True:
                val = TaskBatch2(val, imageSize, task)
            val_xs = val.iloc[:, :imageSize]
            val_ys = val.iloc[:, imageSize:]
            val_ys *= TaskMask

            accTestCurr, acc1out, acc2out, acc3out, hC1sparOut, hC2sparOut, hFC1sparOut =\
                sess.run([accuracy, accuracy1, accuracy2, accuracy3, hConv1sparsity, hConv2sparsity, hFC1sparsity],
                         feed_dict={x: val_xs, ystar: val_ys, switch:switchIn, imageKeepProb: 1, featureKeepProb: 1,
                                    yMask:yMaskIn})
            accTest2[testIndex] = accTestCurr
            accDigit[testIndex] = acc1out
            accStroke[testIndex] = acc2out
            accNoise[testIndex] = acc3out
            activations.append([hC1sparOut+ zeta, hC2sparOut+ zeta, hFC1sparOut + zeta])
            weights.append([Wconv1spar, Wconv2spar, W1spar, W2spar])
            accTest.append(accTestCurr)
            testIndex += 1

            # convergence test
            if iterLoop > 500:
                convTest = np.mean(accTest[-20:]) / np.mean(accTest[-100:])
                if accTest[-1] < accTest[-2] < accTest[-3] < accTest[-4] < accTest[-5]:
                    convTest = 0

            if iterLoop%100 == 0:
                acc1out, acc2out, acc3out, loss1out, loss2out,loss3out = \
                    sess.run([accuracy1, accuracy2, accuracy3, loss1, loss2, loss3],
                             feed_dict={x: val_xs, ystar: val_ys, switch:switchIn, imageKeepProb: 1,
                                        featureKeepProb: 1, yMask:yMaskIn})
                fullAccuracy = sess.run(accuracy, feed_dict= {x: mnist2test.iloc[:, :imageSize],
                                                               ystar: mnist2test.iloc[:, imageSize:],
                                                               switch: [1, 1, 1], imageKeepProb: 1, featureKeepProb: 1,
                                                               yMask:np.ones(10)})
                print("step %d, training accuracy %g, test accuracy %g."  %
                      (iterLoop, batchAccuracy, accTestCurr))
                print("Task accuracies: Digit %g, Stroke width %g, Noise %g, Full %g" % (
                acc1out, acc2out, acc3out, fullAccuracy))
                print("activation sparsity: hConv1 %g, hConv2 %g, hFC1 %g" % (hC1sparOut, hC2sparOut, hFC1sparOut))
                print("weight sparsity: conv1 %g, conv2 %g, Wfc1 %g, Wfc2 %g" % (Wconv1spar, Wconv2spar, W1spar, W2spar))
                print(convTest)
            iterLoop += 1

        # final statistics
        if domAdap is True:
            test = TaskBatch2(mnist2test,imageSize, task)
            test_xs = test.iloc[:, :imageSize]
            test_ys = test.iloc[:, imageSize:]
            test_ys *= TaskMask
            finalAccuracy = sess.run(accuracy, feed_dict={x: test_xs, ystar: test_ys, switch:switchIn,
                 imageKeepProb: 1, featureKeepProb: 1, yMask:yMaskIn})
        else:
            finalAccuracy = sess.run(accuracy,feed_dict=
                {x: mnist2test.iloc[:,:imageSize], ystar: mnist2test.iloc[:,imageSize:] * TaskMask, switch:switchIn,
                 imageKeepProb: 1, featureKeepProb: 1, yMask:yMaskIn})

        print("Version %s, Rep %g, Training Rate %g, Steps %d, Final Training Accuracy %g, Final Test Accuracy %g" %
              (version, repLoop, etaIn, iterLoop, batchAccuracy, finalAccuracy))

        if domAdap is True:
            FisherMatrices = sess.run(Fout, feed_dict={x:batch_xs, switch:switchIn, imageKeepProb: 1,
                                                       featureKeepProb: 1, yMask:yMaskIn})
            for v in range(len(FisherMatrices)):
                FisherMatrices[v] = slin.sqrtm(FisherMatrices[v])
        else:
            FisherMatrices = 0
        # sess.close()
        return finalAccuracy, iterLoop, accTest2, FisherMatrices, activations, weights, [accDigit,accStroke,accNoise],\
               [loss1out,loss2out,loss3out]


    if domAdap is True:
        accuracies = np.zeros([reps, 3])
        curves = np.zeros([reps, 1000, 3])
        curvesDigit = np.zeros([reps, 1000, 3])
        curvesStroke = np.zeros([reps, 1000, 3])
        curvesNoise = np.zeros([reps, 1000, 3])

        steps = np.zeros([reps, 3])
        frechet = np.zeros([reps, 9])
        euclid = np.zeros([reps, 9])
        activationsAll = np.zeros([reps, 3000, 3])
        weightsAll = np.zeros([reps, 3000, 4])

        batchSize = 300
        switchIn = [1,1,1]

        for repLoop in range(reps):
            numberIts = numIts
            etaIn = etaInitial
            lamdaFin = 0
            # create random sets of 3 out of 120 possible combinations
            taskSets = np.random.permutation(10)
            # taskSets = np.arange(0,9)
            taskSets = taskSets[:9].reshape((3, 3))
            FisherMatrices = []
            weightMatrices = []
            sess.run(init)
            for taskLoop in range(3):
                # sess.run(init) # remember to change!!!!!!!!!!!

                yMaskIn = np.zeros(10)
                for zeroLoop in range(3):
                    yMaskIn[taskSets[zeroLoop, taskLoop]] = 1

                with sess.as_default():
                    Wfc2starIn = Wfc2.eval()

                finalAccuracy, iterLoop, curve, FisherOut, activations, weights, TaskCurves = \
                    run(batchSize, numberIts, etaIn, switchIn, taskSets,taskLoop, yMaskIn)[:7]
                accuracies[repLoop, taskLoop] = finalAccuracy
                steps[repLoop, taskLoop] = iterLoop
                curves[repLoop, :, taskLoop] = curve
                curvesDigit[repLoop, :, taskLoop] = TaskCurves[0]
                curvesStroke[repLoop, :, taskLoop] = TaskCurves[1]
                curvesNoise[repLoop, :, taskLoop] = TaskCurves[2]

                FisherMatrices.append(FisherOut)
                weightMatrices.append(sess.run([Wconv1, Wconv2, Wfc1]))
                if taskLoop > 0:
                    taskActivations = np.append(taskActivations, activations, axis=0)
                    taskWeights = np.append(taskWeights, weights, axis=0)
                else:
                    taskActivations = activations
                    taskWeights = weights

                # save embeddings, weight matrices for task
                saveEmbed()
                Wconv1out,Wfc2out = sess.run([Wconv1,Wfc2])
                np.savetxt("wfc2" + version + str(taskLoop) + ".csv", Wfc2out, delimiter=',')

            W1, W2, W3 = weightMatrices

            # calculate Frechet and Euclidean distances
            F1, F2, F3 = FisherMatrices
            for v in range(len(varList) - 1):
                frechet[repLoop, 0 + v * 3] = 0.5 * np.linalg.norm(F1[v] - F2[v])
                frechet[repLoop, 1 + v * 3] = 0.5 * np.linalg.norm(F1[v] - F3[v])
                frechet[repLoop, 2 + v * 3] = 0.5 * np.linalg.norm(F2[v] - F3[v])
                euclid[repLoop, 0 + v * 3] = 0.5 * np.linalg.norm(W1[v] - W2[v])
                euclid[repLoop, 1 + v * 3] = 0.5 * np.linalg.norm(W1[v] - W3[v])
                euclid[repLoop, 2 + v * 3] = 0.5 * np.linalg.norm(W2[v] - W3[v])

            activationsAll[repLoop, :len(taskActivations), :] = taskActivations
            weightsAll[repLoop, :len(taskWeights), :] = taskWeights

        activationsAll[activationsAll == 0] = np.nan
        activationsMean = np.nanmean(activationsAll, axis=0)
        activationsStd = np.nanstd(activationsAll, axis=0)
        activationsOut = np.concatenate((activationsMean, activationsStd), axis=1)

        weightsAll[weightsAll == 0] = np.nan
        weightsMean = np.nanmean(weightsAll, axis=0)
        weightsStd = np.nanstd(weightsAll, axis=0)
        weightsOut = np.concatenate((weightsMean, weightsStd), axis=1)

        curves[curves == 0] = np.nan
        curvesMean = np.nanmean(curves, axis=0)
        std = np.nanstd(curves, axis=0)

        curvesDigit[curvesDigit == 0] = np.nan
        curvesDigitMean = np.nanmean(curvesDigit, axis=0)
        stdDigit = np.nanstd(curvesDigit, axis=0)

        curvesStroke[curvesStroke == 0] = np.nan
        curvesStrokeMean = np.nanmean(curvesStroke, axis=0)
        stdStroke = np.nanstd(curvesStroke, axis=0)

        curvesNoise[curvesNoise == 0] = np.nan
        curvesNoiseMean = np.nanmean(curvesNoise, axis=0)
        stdNoise = np.nanstd(curvesNoise, axis=0)

        frechetMean = np.mean(frechet, axis=0)
        frechetStd = np.std(frechet, axis=0)
        euclidMean = np.mean(euclid, axis=0)
        euclidStd = np.std(euclid, axis=0)

        for v in range(len(varList) - 1):
            print('%s Frechet Distances:' % (varList[v].name))
            print('Task1-Task2: %g' % (frechetMean[0 + v * 3]))
            print('Task1-Task3: %g' % (frechetMean[1 + v * 3]))
            print('Task2-Task3: %g' % (frechetMean[2 + v * 3]))

            print('%s Euclidean Distances:' % (varList[v].name))
            print('Task1-Task2: %g' % (euclidMean[0 + v * 3]))
            print('Task1-Task3: %g' % (euclidMean[1 + v * 3]))
            print('Task2-Task3: %g' % (euclidMean[2 + v * 3]))


        np.savetxt("frechet" + version + ".csv", np.stack((frechetMean, frechetStd), axis=-1), delimiter=',')
        np.savetxt("euclid" + version + ".csv", np.stack((euclidMean, euclidStd), axis=-1), delimiter=',')
        np.savetxt("curves" + version + ".csv", np.concatenate((curvesMean, std), axis=1),
                   delimiter=',', header='task1,task2,task3,std1,std2,std3')
        np.savetxt("curvesDigit" + version + ".csv", np.concatenate((curvesDigitMean, stdDigit), axis=1),
                   delimiter=',', header='task1,task2,task3,std1,std2,std3')
        np.savetxt("curvesStroke" + version + ".csv", np.concatenate((curvesStrokeMean, stdStroke), axis=1),
                   delimiter=',', header='task1,task2,task3,std1,std2,std3')
        np.savetxt("curvesNoise" + version + ".csv", np.concatenate((curvesNoiseMean, stdNoise), axis=1),
                   delimiter=',', header='task1,task2,task3,std1,std2,std3')

        np.savetxt("activationSpar" + version + ".csv", np.concatenate((activationsMean, activationsStd), axis=1),
                   delimiter=',', header='hconv1,hconv2,hfc1,std1,std2,std3')
        np.savetxt("weightSpar" + version + ".csv", np.concatenate((weightsMean, weightsStd), axis=1),
                   delimiter=',', header='wconv1,wconv2,wfc1,wfc2,std1,std2,std3, std4')

        meanAccuracy, stdAccuracy, meanTaskAccuracies,stdTaskAccuracies = [curvesMean,std,activationsOut,weightsOut]

    else:
        accuracies = np.zeros([reps])
        taskAccuracies = np.zeros([reps,3])
        taskLosses = np.zeros([reps,3])
        curves = np.zeros([reps,1000])
        steps = np.zeros([reps])

        for repLoop in range(reps):
            sess.run(init)
            etaIn = etaInitial
            with sess.as_default():
                Wfc2starIn = Wfc2.eval()
            finalAccuracy, iterLoop, curve, FisherOut, activations, weights, taskAccs, taskLoss \
                = run(batchSize, numIts, etaIn, switchIn, np.zeros((3,3)),1, np.ones(10))
            saveEmbed()
            accuracies[repLoop] = finalAccuracy
            taskAccuracies[repLoop,:] = np.array(taskAccs)[:,iterLoop-1]
            taskLosses[repLoop,:] = taskLoss
            steps[repLoop] = iterLoop
            curves[repLoop, :] = curve

        meanAccuracy = np.mean(accuracies)
        stdAccuracy = np.std(accuracies)
        meanTaskAccuracies = np.mean(taskAccuracies,axis=0)
        stdTaskAccuracies = np.std(taskAccuracies,axis=0)
        meanTaskLosses = np.mean(taskLosses,axis=0)
        stdTaskLosses = np.std(taskLosses,axis=0)
        steps = np.mean(steps)

        print("hidUnits %d, rep %d, Average stats: Steps %d, Final Accuracy %g" % (hidUnitLoop, repLoop, steps, meanAccuracy))
        print("Average task accuracies: Digit %g+/-%g, Stroke width %g+/-%g, Noise %g+/-%g" %
              (meanTaskAccuracies[0],stdTaskAccuracies[0],meanTaskAccuracies[1],stdTaskAccuracies[1],meanTaskAccuracies[2],
               stdTaskAccuracies[2]))
        print("Average task loss: Digit %g+/-%g, Stroke width %g+/-%g, Noise %g+/-%g" %
              (meanTaskLosses[0],stdTaskLosses[0],meanTaskLosses[1],stdTaskLosses[1],meanTaskLosses[2],stdTaskLosses[2]))

    return meanAccuracy, stdAccuracy, meanTaskAccuracies, stdTaskAccuracies

hidList = [32]#[32,64,128,256,512]

results = np.zeros([7,len(hidList)])
stds = np.zeros([6,len(hidList)])
resultInd = 0
for hidUnitLoop in hidList:
    numParam = hidUnitLoop * (7 * 7 * convFilters +  tasks) + 5 * 5 * 1 * convFilters + 3 * 3 * convFilters * convFilters
    meanAccuracy, stdAccuracy, meanTaskAccuracies, stdTaskAccuracies = model([1,1,1],hidUnitLoop)

    # to average over many runs
    # results[0,resultInd] = numParam
    # results[1:4,resultInd] = meanTaskAccuracies
    # stds[:3,resultInd] = stdTaskAccuracies
    # taskSwitch = [1, 0, 0]
    # for singleLoop in range(3):
    #     meanAccuracy, stdAccuracy, meanTaskAccuracies, stdTaskAccuracies = model(taskSwitch, hidUnitLoop)
    #     results[4+singleLoop, resultInd] = meanAccuracy
    #     stds[3+singleLoop, resultInd] = stdAccuracy
    #     taskSwitch = np.roll(taskSwitch, 1)
    # resultInd += 1
