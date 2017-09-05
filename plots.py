# Graeme Blyth 2017
# uncomment relevant section to generate data

import numpy as np
import matplotlib.pyplot as plt
from plot_imagegrid import plot_imagegrid


def scale(array):
    min = np.amin(array)
    array -= min
    max = np.amax(array)
    array /= max
    return array

# path = '/Users/graeme/PycharmProjects/downloads/new/private3/'
path = '/Users/graeme/PycharmProjects/TL/'

#save or display plots of final weight matrix, manually enter which stage of optimisation
version1 = 'DropFull1'
version2 = 'l22'
zeroThresh = 1e-2

Wfc2out = np.loadtxt(path+ 'wfc2' + version1 + ".csv", delimiter=',')
Wfc2out2 = np.loadtxt(path+'wfc2' + version2 + ".csv", delimiter=',')
# Wfc2out -= Wfc2out2
Wfc2out[Wfc2out<zeroThresh] = 0
spar = 1 - (np.count_nonzero(Wfc2out) / Wfc2out.size)
print(spar)
#
plt.imshow(scale(Wfc2out), aspect='auto')  # .astype('float32'))
plt.gray()
plt.show()
# plt.savefig(path+ 'wfc2' + version1 +'.png')

# Wconv1out = np.load(path+'wconv1' + version1 + ".npy")
# # Wconv1out2 = np.load(path+'wconv1' + version2 + ".npy")
# # Wconv1out -= Wconv1out2
# fig = plot_imagegrid(np.moveaxis(Wconv1out, -1, 0),scaling=True)
#
# plt.show()



#fetch training curves
# path2 = '/Users/graeme/PycharmProjects/download/new/private3/curves'
# curves0 = np.round_(np.loadtxt(path2 + 'noRegs' + ".csv", delimiter=',')*100,1)
# curves1 = np.round_(np.loadtxt(path2 + 'l1' + ".csv", delimiter=',')*100,1)
# curves2 = np.round_(np.loadtxt(path2 + 'l2' + ".csv", delimiter=',')*100,1)
# curves3 = np.round_(np.loadtxt(path2 + 'drop' + ".csv", delimiter=',')*100,1)
#
#
# for l in range(3):
#     print('new table')
#     print("None & %g $\pm$ %g & %g $\pm$ %g & %g $\pm$ %g & %g $\pm$ %g"
#           %(curves0[1,0+l],curves0[1,3+l],curves0[20,0+l],curves0[20,3+l],
#             curves0[40,0+l],curves0[40,3+l],curves0[500,0+l],curves0[500,3+l]) + r'\\')
#     print("$\ell_1$ & %g $\pm$ %g & %g $\pm$ %g & %g $\pm$ %g & %g $\pm$ %g"
#           %(curves1[1,0+l],curves1[1,3+l],curves1[20,0+l],curves1[20,3+l],
#             curves1[40,0+l],curves1[40,3+l],curves1[500,0+l],curves1[500,3+l]) + r'\\')
#     print("$\ell_2$ & %g $\pm$ %g & %g $\pm$ %g & %g $\pm$ %g & %g $\pm$ %g"
#           %(curves2[1,0+l],curves2[1,3+l],curves2[20,0+l],curves2[20,3+l],
#             curves2[40,0+l],curves2[40,3+l],curves2[500,0+l],curves2[500,3+l]) + r'\\')
#     print("$Dropout$ & %g $\pm$ %g & %g $\pm$ %g & %g $\pm$ %g & %g $\pm$ %g"
#       % (curves3[1, 0 + l], curves3[1, 3 + l], curves3[20, 0 + l], curves3[20, 3 + l],
#          curves3[40, 0 + l], curves3[40, 3 + l], curves3[500, 0 + l], curves3[500, 3 + l]) + r'\\')


#fetch performance mnist2
# path2 = '/Users/graeme/PycharmProjects/download/new/private3/curves'
# version = 'Noisemnist2'
# curves0 = np.round_(np.loadtxt(path2 + version + 'noRegs' + ".csv", delimiter=',')*100,1)
# curves1 = np.round_(np.loadtxt(path2 + version + 'l1' + ".csv", delimiter=',')*100,1)
# curves2 = np.round_(np.loadtxt(path2 + version + 'l2' + ".csv", delimiter=',')*100,1)
# curves3 = np.round_(np.loadtxt(path2 + version + 'Ortho' + ".csv", delimiter=',')*100,1)
#
#
#
# for l in range(3):
#     print('new table')
#     print("None & %g $\pm$ %g & %g $\pm$ %g & %g $\pm$ %g & %g $\pm$ %g"
#           %(curves0[0,0+l],curves0[0,3+l],curves0[19,0+l],curves0[19,3+l],
#             curves0[39,0+l],curves0[39,3+l],curves0[499,0+l],curves0[499,3+l]) + r'\\')
#     print("$\ell_1$ & %g $\pm$ %g & %g $\pm$ %g & %g $\pm$ %g & %g $\pm$ %g"
#           %(curves1[0,0+l],curves1[0,3+l],curves1[19,0+l],curves1[19,3+l],
#             curves1[39,0+l],curves1[39,3+l],curves1[499,0+l],curves1[499,3+l]) + r'\\')
#     print("$\ell_2$ & %g $\pm$ %g & %g $\pm$ %g & %g $\pm$ %g & %g $\pm$ %g"
#           %(curves2[0,0+l],curves2[0,3+l],curves2[19,0+l],curves2[19,3+l],
#             curves2[39,0+l],curves2[39,3+l],curves2[499,0+l],curves2[499,3+l]) + r'\\')
#     print("$Ortho$ & %g $\pm$ %g & %g $\pm$ %g & %g $\pm$ %g & %g $\pm$ %g"
#           % (curves3[0, 0 + l], curves3[0, 3 + l], curves3[19, 0 + l], curves3[19, 3 + l],
#              curves3[39, 0 + l], curves3[39, 3 + l], curves3[499, 0 + l], curves3[499, 3 + l]) + r'\\')



#fetch frechet
# path2 = '/Users/graeme/PycharmProjects/download/new/private3/frechet'
# frechet = []
# frechetMean = np.zeros([3,2])
# frechet.append( np.loadtxt(path2 + 'noRegs' + ".csv", delimiter=','))
# frechet.append( np.loadtxt(path2 + 'l1' + ".csv", delimiter=','))
# frechet.append( np.loadtxt(path2 + 'l2' + ".csv", delimiter=','))
# frechet.append( np.loadtxt(path2 + 'drop' + ".csv", delimiter=','))
#
#
#
# labels = ['a','b','c','d']
# varList = ['Wconv1','Wconv2','Wfc1']
# for methods in range(4):
#     index = 0
#     for v in range(3):
#         mean = np.mean(frechet[methods][index:index+3,0])
#         std = np.sqrt(np.mean(frechet[methods][index:index+3,1]**2))
#         index+=3
#         print('(%s, %g)+-(%g,%g)[%s]' % (varList[v], mean, std, std, labels[methods]))

#fetch frechet mnist2
# path2 = '/Users/graeme/PycharmProjects/download/new/private3/frechetmnist2'
# frechet = []
# frechetMean = np.zeros([3,2])
# frechet.append( np.loadtxt(path2 + 'noRegs' + ".csv", delimiter=','))
# frechet.append( np.loadtxt(path2 + 'l1' + ".csv", delimiter=','))
# frechet.append( np.loadtxt(path2 + 'l2' + ".csv", delimiter=','))
# frechet.append( np.loadtxt(path2 + 'ortho' + ".csv", delimiter=','))
#
#
# labels = ['a','b','c','d']
# varList = ['Wconv1','Wconv2','Wfc1']
# for methods in range(4):
#     index = 0
#     for v in range(3):
#         mean = np.mean(frechet[methods][index:index+3,0])
#         std = np.sqrt(np.mean(frechet[methods][index:index+3,1]**2))
#         index+=3
#         print('(%s, %g)+-(%g,%g)[%s]' % (varList[v], mean, std, std, labels[methods]))
#

#fetch euclid
# path2 = '/Users/graeme/PycharmProjects/download/new/private3/euclid'
# euclid = []
# euclidMean = np.zeros([3,2])
# euclid.append( np.loadtxt(path2 + 'l1' + ".csv", delimiter=','))
# euclid.append( np.loadtxt(path2 + 'l2' + ".csv", delimiter=','))
# euclid.append( np.loadtxt(path2 + 'noRegs' + ".csv", delimiter=','))
# euclid.append( np.loadtxt(path2 + 'drop' + ".csv", delimiter=','))
#
#
# labels = ['a','b','c','d']
# varList = ['Wconv1','Wconv2','Wfc1']
# for methods in range(4):
#     index = 0
#     for v in range(3):
#         mean = np.mean(euclid[methods][index:index+3,0])
#         std = np.sqrt(np.mean(euclid[methods][index:index+3,1]**2))
#         index+=3
#         print('(%s, %g)+-(%g,%g)[%s]' % (varList[v], mean, std, std, labels[methods]))
