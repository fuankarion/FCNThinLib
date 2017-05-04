import matplotlib
import sys
matplotlib.use('Agg')
from pylab import *

caffeRoot='/home/jcleon/Software/caffe/python'
sys.path.insert(0, caffeRoot)

import caffe #Caffe Lib

from coreTrainFCN import trainNetworkLog

caffe.set_device(1)
caffe.set_mode_gpu()

#Actual Params
solverPath = './net/solver.prototxt'
importModelPath = './net/fcn_alexnet.caffemodel'

#Solver data 
solver = caffe.get_solver(solverPath)
solver.net.copy_from(importModelPath)

#Training Config (same as in solver)
batchesForTraining = 294
batchesUntillStep = 15;
maxSteps = 3;
niter = batchesForTraining * batchesUntillStep * maxSteps
test_iters = 32

#Log Data
targetLogFile = './net/log.txt'
trainNetworkLog(solver, niter, batchesForTraining, targetLogFile, test_iters, 6, 5)

