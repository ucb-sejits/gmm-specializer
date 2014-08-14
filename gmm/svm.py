"""
specializer svm

follows algorithm from Catanzaro et al.
"""
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
import pycl as cl
import os
import numpy as np
from math import *
from random import random
from ctree.ocl.nodes import *
from ctree.templates.nodes import *
from ctree.c.nodes import *
from ctypes import *
from numpy.ctypeslib import as_array
from profilehooks import *
import logging
import time
from collections import OrderedDict

# logging.basicConfig(level=20)
class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start
class OclTrainFunction(ConcreteSpecializedFunction):
    def __init__(self):
        self.context = cl.clCreateContextFromType()
        self.queue = cl.clCreateCommandQueue(context=self.context)#, properties=cl.cl_command_queue_properties.CL_QUEUE_PROFILING_ENABLE)
        self.device = self.queue.device

    def finalize(self, program, tree, entry_name):
        self.program = program

        self.init = program["initializeArrays"]
        self.step1 = program["doFirstStep"]
        self.foph1 = program["firstOrderPhaseOne"]
        self.foph2 = program["firstOrderPhaseTwo"]
        self.soph1 = program["secondOrderPhaseOne"]
        self.soph2 = program["secondOrderPhaseTwo"]
        self.soph3 = program["secondOrderPhaseThree"]
        self.soph4 = program["secondOrderPhaseFour"]



        entry_type = CFUNCTYPE(c_int, c_int, POINTER(c_float), POINTER(c_int), c_float, c_float, c_float, c_float, c_int, c_int, c_int,

                                c_float, c_float, c_float, c_size_t, c_size_t,
                                c_int, c_size_t, c_size_t, c_size_t, c_size_t,
                                c_int, c_size_t, c_size_t, c_size_t, c_size_t,
                                c_int, c_size_t, c_size_t, c_size_t, c_size_t,

                                cl.cl_mem, cl.cl_mem, cl.cl_mem, cl.cl_mem, cl.cl_mem, cl.cl_mem,
                                cl.cl_mem, cl.cl_mem, cl.cl_mem, cl.cl_mem, cl.cl_mem,
                                cl.cl_mem, cl.cl_mem, cl.cl_mem, cl.cl_mem, cl.cl_mem, cl.cl_mem,

                                cl.cl_command_queue, cl.cl_kernel, cl.cl_kernel, cl.cl_kernel, cl.cl_kernel,
                                cl.cl_kernel, cl.cl_kernel, cl.cl_kernel, cl.cl_kernel,

                                POINTER(c_float), POINTER(c_int),
                                POINTER(c_int), POINTER(POINTER(c_float)), POINTER(POINTER(c_float)))

        self._c_function = self._compile(entry_name, tree, entry_type)
        return self

    def __call__(self, *args):
        cacheSize = args[1]
        cacheSizeInFloats = (int)(cacheSize/sizeof(c_float))
        args = args[2:]
        nPoints, dFeatures = args[0].shape


        localInitSize = self.init.work_group_size(self.device)
        numGroups_init = nPoints/localInitSize if (nPoints % localInitSize == 0) else nPoints/localInitSize + 1
        globalInitSize = numGroups_init * localInitSize

        localFoph1Size = self.foph1.work_group_size(self.device)
        numGroups_foph1 = nPoints/localFoph1Size if (nPoints % localFoph1Size == 0) else nPoints/localFoph1Size + 1
        globalFoph1Size = numGroups_foph1 * localFoph1Size

        localFoph2Size = self.foph2.work_group_size(self.device)
        globalFoph2Size = localFoph2Size

        localSoph1Size = self.soph1.work_group_size(self.device)
        numGroups_soph1 = nPoints/localSoph1Size if (nPoints % localSoph1Size == 0) else nPoints/localSoph1Size + 1
        globalSoph1Size = numGroups_soph1 * localSoph1Size

        localSoph2Size = self.soph2.work_group_size(self.device)
        globalSoph2Size = localSoph2Size

        localSoph3Size = self.soph3.work_group_size(self.device)
        numGroups_soph3 = nPoints/localSoph3Size if (nPoints % localSoph3Size == 0) else nPoints/localSoph3Size + 3
        globalSoph3Size = numGroups_soph3 * localSoph3Size

        localSoph4Size = self.soph4.work_group_size(self.device)
        globalSoph4Size = localSoph4Size
        # print "Init .... Local: %d, Num Groups: %d, Global: %d" % (localInitSize, numGroups_init, globalInitSize)
        # print "Foph1 ... Local: %d, Num Groups: %d, Global: %d" % (localFoph1Size, numGroups_foph1, globalFoph1Size)
        # print "Foph2 ... Local & Global: %d" % (localFoph2Size)
        # print "Soph1 ... Local: %d, Num Groups: %d, Global: %d" % (localSoph1Size, numGroups_soph1, globalSoph1Size)
        # print "Soph2 ... Local & Global: %d" % (localSoph2Size)
        # print "Soph3 ... Local: %d, Num Groups: %d, Global: %d" % (localSoph3Size, numGroups_soph3, globalSoph3Size)
        # print "Soph4 ... Local & Global: %d" % (localSoph4Size)

        #create buffers from input
        d_input_data, evt = cl.buffer_from_ndarray(self.queue, args[0], blocking = False)
        d_input_data_colmajor, evt = cl.buffer_from_ndarray(self.queue, args[0].T, blocking = False)
        d_labels, evt = cl.buffer_from_ndarray(self.queue, args[1], blocking = False)
        args = (args[0].ctypes.data_as(POINTER(c_float)), args[1].ctypes.data_as(POINTER(c_int))) + args[2:]

        # temporary numpy arrays
        iArray = np.zeros(nPoints, dtype=np.float32)

        reduceIntsFO = np.zeros(numGroups_foph1, dtype = np.int32)
        reduceFloatsFO = np.zeros(numGroups_foph1, dtype = np.float32)

        reduceIntsSO1 = np.zeros(numGroups_soph1, dtype = np.int32)
        reduceFloatsSO1 = np.zeros(numGroups_soph1, dtype = np.float32)

        reduceIntsSO3 = np.zeros(numGroups_soph3, dtype = np.int32)
        reduceFloatsSO3 = np.zeros(numGroups_soph3, dtype = np.float32)

        results = np.zeros(8, dtype= np.float32)
        cache = np.zeros(cacheSizeInFloats, dtype=np.float32)

        # new buffers from scratch
        d_trainingAlpha, evt = cl.buffer_from_ndarray(self.queue, iArray, blocking = False)
        d_kernelDiag, evt = cl.buffer_from_ndarray(self.queue, iArray, blocking = False)
        d_F, evt = cl.buffer_from_ndarray(self.queue, iArray, blocking = False)
        d_cache, evt =  cl.buffer_from_ndarray(self.queue, cache, blocking = False)

        d_highFsFO, evt = cl.buffer_from_ndarray(self.queue, reduceFloatsFO, blocking = False)
        d_highIndicesFO, evt = cl.buffer_from_ndarray(self.queue, reduceIntsFO, blocking = False)

        d_lowFsFO, evt = cl.buffer_from_ndarray(self.queue, reduceFloatsFO, blocking = False)
        d_lowIndicesFO, evt = cl.buffer_from_ndarray(self.queue, reduceIntsFO, blocking = False)

        d_highFsSO1, evt = cl.buffer_from_ndarray(self.queue, reduceFloatsSO1, blocking = False)
        d_highIndicesSO1, evt = cl.buffer_from_ndarray(self.queue, reduceIntsSO1, blocking = False)

        d_lowFsSO3, evt = cl.buffer_from_ndarray(self.queue, reduceFloatsSO3, blocking = False)
        d_lowIndicesSO3, evt = cl.buffer_from_ndarray(self.queue, reduceIntsSO3, blocking = False)
        d_deltaFsSO3, evt = cl.buffer_from_ndarray(self.queue, reduceFloatsSO3, blocking = False)

        d_results, evt = cl.buffer_from_ndarray(self.queue, results, blocking = False)

        #return values
        rho = c_float()
        nSV = c_int()
        iterations = c_int()
        signedAlpha = pointer(c_float())
        supportVectors = pointer(c_float())

        args = (cacheSizeInFloats,)+args
        #add args
        args += (
                # sizes
                localInitSize, globalInitSize,
                numGroups_foph1, localFoph1Size, globalFoph1Size,
                localFoph2Size , globalFoph2Size,
                numGroups_soph1, localSoph1Size, globalSoph1Size,
                localSoph2Size, globalSoph2Size,
                numGroups_soph3, localSoph3Size, globalSoph3Size,
                localSoph4Size, globalSoph4Size,
                # buffers
                d_input_data, d_input_data_colmajor, d_labels, d_trainingAlpha, d_kernelDiag, d_F,
                d_highFsFO, d_highIndicesFO, d_lowFsFO, d_lowIndicesFO,
                d_highFsSO1, d_highIndicesSO1, d_lowFsSO3, d_lowIndicesSO3, d_deltaFsSO3,
                d_results, d_cache,
                # Ocl queue and kernels
                self.queue, self.init, self.step1, self.foph1, self.foph2,
                self.soph1, self.soph2, self.soph3, self.soph4,
                # return pointers
                byref(rho), byref(nSV), byref(iterations), byref(signedAlpha), byref(supportVectors))

        with Timer() as cfunc:
            err = self._c_function(*args)
        print "Actual function took %.6f" % cfunc.interval
        return rho.value, nSV.value, iterations.value, \
               as_array(supportVectors,shape=(nSV.value,dFeatures)),\
               as_array(signedAlpha,shape=(nSV.value,))

class OclTrain(LazySpecializedFunction):

    def args_to_subconfig(self, args):
        return (args[0])
        # conf = ()
        # for arg in args[:2]:
        #     conf += ((arg.dtype, arg.ndim, arg.shape),)
        # return conf
    def transform(self, tree, program_config):


        kernelFunc = program_config[0]
        kernelPath = os.path.join(os.getcwd(), "..", "templates","trainingkernels.tmpl.c")
        kernelInserts = {
            "kernelFunc": SymbolRef(kernelFunc),
        }
        kernel = OclFile("training_kernel", [
            FileTemplate(kernelPath, kernelInserts)
        ])

        wrapperPath = os.path.join(os.getcwd(), "..", "templates","ocltrain.tmpl.c")
        wrapperInserts = {
            "kernel_path": kernel.get_generated_path_ref(),
            "kernelFunc": SymbolRef(kernelFunc)
        }
        wrapper = CFile("train", [
            FileTemplate(wrapperPath, wrapperInserts)
        ])
        fn = OclTrainFunction()
        program = cl.clCreateProgramWithSource(fn.context, kernel.codegen()).build()
        return fn.finalize(program, Project([kernel, wrapper]),"train")


class SVM(object):

    def setParams(self, kernel_type, nPoints, dFeatures, gamma = None, coef0 = None, degree = None):
        self.kernelFuncs = {'linear': self.linear,
                            'gaussian': self.gaussian,
                            'polynomial': self.polynomial,
                            'sigmoid': self.sigmoid}
        self.kernelFuncsSelf = {'linear': self.linearSelf,
                            'gaussian': self.gaussianSelf,
                            'polynomial': self.polynomialSelf,
                            'sigmoid': self.sigmoidSelf}
        if kernel_type not in self.kernelFuncs:
            raise Exception( "Unsupported kernel type. Please try one of the following: \
                  'linear', 'gaussian', 'polynomial', 'sigmoid'")

        self.kernelFunc = self.kernelFuncs[kernel_type]
        self.kernelFuncSelf = self.kernelFuncsSelf[kernel_type]
        #default params
        if gamma is None:
            gamma = 1.0/dFeatures
        if coef0 is None:
            coef0 = 0.0
        if degree is None:
            degree = 3

        if gamma <= 0 or coef0 < 0 or degree < 1.0:
            raise Exception( "Invalid parameters")

        self.gamma = - gamma
        self.coef0 = coef0
        self.degree = (int)(degree)
        self.kernel_type = kernel_type
        self.params = {"gamma": - gamma, "coef0": coef0, "degree": degree, "dFeatures": dFeatures}

    def __init__(self):
        self.pythonOnly = False
        self.nPoints = 0
        self.dFeatures = 0

        self.input_data = None
        self.labels = None

        self.support_vectors = None
        self.signed_alpha = None
        self.rho = None
        self.nSV = 0
        self.iterations = 0
        self.OclTrain = OclTrain(None)


    def train(self, input_data, labels, kernel_type, cacheSizeMB = 100,
              gamma = None, coef0 = None, degree = None,
              heuristicMethod = None, tolerance = None, cost = None, epsilon = None, pythonOnly = False):
        self.pythonOnly = pythonOnly

        self.nPoints = input_data.shape[0]
        self.dFeatures = input_data.shape[1]
        self.input_data = input_data
        self.labels = labels
        self.result = np.zeros(8, dtype=np.float32)
        self.cacheSize = cacheSizeMB * 1000000

        self.setParams(kernel_type,self.nPoints, self.dFeatures,gamma,coef0,degree)
        self.heuristic = heuristicMethod if heuristicMethod is not None else 2 #random
        self.cost = cost if cost is not None else 10.0
        self.tolerance = tolerance if tolerance is not None else 1e-3
        self.epsilon = epsilon if epsilon is not None else 1e-5
        self.Ce = self.cost - self.tolerance

        if self.pythonOnly == True:
            self.trainFunc = self.pytrain
        else:
            self.trainFunc = self.specializedTrain

        result = self.trainFunc(self.input_data,self.labels,
                     self.epsilon, self.Ce, self.cost, self.tolerance,
                     self.heuristic, self.nPoints, self.dFeatures, self.params)
        self.rho, self.nSV, self.iterations, self.support_vectors, self.signed_alpha = result

    def pytrain(self, input_data, labels, epsilon, Ce, cost, tolerance,
                heuristic, nPoints, dFeatures, params):
        #initialize
        iLow =None
        iHigh = None
        for i in range(nPoints):
            label = labels[i]
            if iLow is None and label == -1:
                iLow = i
            elif iHigh is None and label == 1:
                iHigh = i
        trainingAlpha = np.zeros(self.nPoints, dtype=np.float32)
        F = np.empty(nPoints, dtype= np.float32)
        progress = Controller(2.0, heuristic, 64, nPoints)
        bLow = 1.0
        bHigh = -1.0
        gap = bHigh - bLow
        kernelDiag = np.zeros(self.nPoints, dtype=np.float32)
        for i in range(nPoints):
            kernelDiag[i] = self.kernelFuncSelf(input_data[i], params)
            F[i] = -labels[i]

        #region First step/half-iteration
        # Initializes 2 values in the currently-zero trainingAlpha array

        # copied from below
        # save old alphas
        alpha2old = trainingAlpha[iLow]
        alpha1old = trainingAlpha[iHigh]

        alphadiff = alpha2old - alpha1old
        lowLabel = labels[iLow]
        sign = labels[iHigh]*lowLabel

        # find lower and upper bounds L and H
        if (sign < 0):
            if(alphadiff < 0):
                L = 0
                H = cost + alphadiff
            else:
                L = alphadiff
                H = cost
        else:
            alphaSum = alpha2old + alpha1old
            if alphaSum < cost:
                L = 0
                H = alphaSum
            else:
                L = cost - alphaSum
                H = cost

        # compute and clip alpha2new but only if eta is positive, i.e. second derivative is negative
        eta = kernelDiag[iLow] + kernelDiag[iHigh]
        phiAB = self.kernelFunc(input_data[iHigh], input_data[iLow], params)
        eta -= 2.0 * phiAB

        if eta > 0:
            #compute
            alpha2new = alpha2old + labels[iLow]*gap/eta
            #clip
            if (alpha2new < L):
                alpha2new = L
            elif(alpha2new > H):
                alpha2new = H
        else: # alpha2new can now only assume endpoints or alpha2old (this is rare)
            slope = lowLabel * gap
            delta = slope * (H-L)
            if delta > 0:
                if slope > 0:
                    alpha2new = H
                else:
                    alpha2new = L
            else:
                alpha2new = alpha2old

        alpha2diff = alpha2new - alpha2old
        alpha1diff = -sign * alpha2diff
        alpha1new = alpha1old + alpha1diff
        trainingAlpha[iHigh] = alpha1new
        trainingAlpha[iLow] = alpha2new
        #endregion

        #To clear things up, trainingAlpha[self.iLow] -> alpha2 and trainingAlpha[self.iHigh] ->alpha1
        #endregion
        #Main Loop
        iteration = 0
        while True:
            if bLow <= bHigh + 2*tolerance:
                break #Convergence!
            if (iteration & 0x7f) == 0:
                print ("Iteration: {}, gap: {}").format(iteration, bLow - bHigh)
            if heuristic == 0:
                firstOrder = True
            else:
                firstOrder = False

            # Update F
            for i in range(nPoints):
                F[i] += labels[iHigh]*alpha1diff*self.kernelFunc(input_data[i],input_data[iHigh], params) +\
                        labels[iLow]*alpha2diff*self.kernelFunc(input_data[i],input_data[iLow], params)

            # Note: bHigh and bLow are computed using mapReduce in the CUDA version.

            #region compute bHigh and self.iHigh. self.iHigh = argMin(F[i]: i in I_High), bHigh = min(F[i]: i in I_High)
            bHigh = None

            for i in range(nPoints):
                alpha = trainingAlpha[i]
                label = labels[i]
                #i.e. if i in I_High
                if (epsilon < alpha < Ce) or \
                        (label > 0 and alpha <= self.epsilon) or \
                        (label < 0 and alpha >= self.Ce):
                    f = F[i]
                    if bHigh is None or f < bHigh:
                        bHigh = f
                        iHigh = i
            #endregion

            # region compute bLow and iHigh
            #  bLow = max(F[i]: i in I_Low); compute self.iLow = argMax(F[i]: i in I_Low)
            maxDeltaF = None
            bLow = None
            for i in range(nPoints):
                alpha = trainingAlpha[i]
                label = labels[i]
                # i.e., if i in I_Low
                if(label < 0 and alpha < Ce) or \
                        (label > 0 and alpha >epsilon):
                    f = F[i]
                    if bLow is None or f > bLow :
                        bLow = f
                        if firstOrder:
                            iLow = i
                    if not firstOrder:  # second order
                        beta = bHigh - F[i]
                        if beta <= epsilon:
                            eta = kernelDiag[iHigh] + kernelDiag[i]
                            phiAB = self.kernelFunc(input_data[iHigh], input_data[i], params)
                            eta -= 2.0 * phiAB
                            if eta <= 0: eta = epsilon
                            deltaF = beta * beta /eta
                            if maxDeltaF is None or deltaF > maxDeltaF:
                                iLow = i
                                maxDeltaF = deltaF
            #endregion

            #region Update alphas

            #save previous alphas
            gap = bHigh - bLow
            alpha2old = trainingAlpha[iLow]
            alpha1old = trainingAlpha[iHigh]
            alphadiff = alpha2old - alpha1old
            lowLabel = labels[iLow]
            sign = labels[iHigh]*lowLabel

            # find lower and upper bounds L and H
            if (sign < 0):
                if(alphadiff < 0):
                    L = 0
                    H = cost + alphadiff
                else:
                    L = alphadiff
                    H = cost
            else:
                alphaSum = alpha2old + alpha1old
                if alphaSum < cost:
                    L = 0
                    H = alphaSum
                else:
                    L = cost - alphaSum
                    H = cost

            # compute and clip alpha2new but only if eta is positive, i.e. second derivative is negative
            eta = kernelDiag[iLow] + kernelDiag[iHigh]
            phiAB = self.kernelFunc(input_data[iHigh], input_data[iLow], params)
            eta -= 2.0 * phiAB
            if eta > 0:
                #compute
                alpha2new = alpha2old + lowLabel*(gap)/eta
                #clip
                if (alpha2new < L):
                    alpha2new = L
                elif(alpha2new > H):
                    alpha2new = H
            else: # else alpha2new can only assume endpoints or alpha2old (this is rare)
                slope = lowLabel * gap
                delta = slope * (H-L)
                if delta > 0:
                    if slope > 0:
                        alpha2new = H
                    else:
                        alpha2new = L
                else:
                    alpha2new = alpha2old

            alpha2diff = alpha2new - alpha2old
            alpha1diff = -sign*alpha2diff
            alpha1new = alpha1old + alpha1diff
            trainingAlpha[iHigh] = alpha1new
            trainingAlpha[iLow] = alpha2new
            #print "iLow:%d, iHigh:%d, bLow:%.6f, bHigh:%.6f" % (iLow, iHigh, bLow, bHigh)

            #endregion
            iteration += 1

        # save results
        print "INFO: %d iterations" %(iteration)
        print "INFO: bLow: %f, bHigh %f" %(bLow, bHigh)
        rho = (bHigh + bLow)/2
        nSV = 0
        for k in range(nPoints):
            if trainingAlpha[k] > epsilon:
                nSV += 1
        support_vectors = np.empty((nSV,dFeatures),dtype = np.float32)
        signed_alpha = np.empty(nSV, dtype=np.float32)
        index = 0
        for k in range(nPoints):
            if trainingAlpha[k] > epsilon:
                support_vectors[index] = input_data[k]
                signed_alpha[index] = labels[k] * trainingAlpha[k]
                index += 1
        return rho, nSV, iteration, support_vectors, signed_alpha

    def specializedTrain(self, *args):
        params = args[-1]
        paramA = params["gamma"]
        paramB = params["coef0"]
        paramC = params["degree"]
        args = (self.kernel_type,self.cacheSize,) + args[:-1] + (paramA, paramB, paramC)
        return self.OclTrain(*args)

    def classify(self, points_in):
        numPoints = points_in.shape[0]
        print 'Classification started: {} points to classify.'.format(numPoints)

        labels_out = np.empty(numPoints, dtype = np.int32)
        sum = 0
        for i in range(numPoints):
            if i % 10000 == 0:
                print '{} points classified'.format(i)
            sum = 0
            for j in range(self.nSV):
                sum += self.signed_alpha[j] * self.kernelFunc(points_in[i], self.support_vectors[j] , self.params)
            if sum - self.rho > 0:
                labels_out[i] = 1
            else:
                labels_out[i] = -1
        return labels_out

    def linearSelf(self, vecA, params):
        accumulant = 0.0
        for d in range(params["dFeatures"]):
            accumulant += vecA[d] * vecA[d]
        return accumulant

    def linear(self, vecA, vecB, params):
        accumulant = 0.0
        for d in range(params["dFeatures"]):
            accumulant += vecA[d] * vecB[d]
        return accumulant

    def gaussianSelf(self, vecA, params):
        return 1.0

    def gaussian(self, vecA, vecB, params):
        accumulant = 0.0
        for d in range(params["dFeatures"]):
            diff = vecA[d] - vecB[d]
            accumulant += diff * diff
        return exp(- params["gamma"] * accumulant)

    def polynomialSelf(self, vecA, params):
        accumulant = 0.0
        for d in range(params["dFeatures"]):
            accumulant += vecA[d] * vecA[d]
        return (params["gamma"] * accumulant + params["coef0"]) ** params["degree"]

    def polynomial(self, vecA, vecB, params):
        accumulant = 0.0
        for d in range(params["dFeatures"]):
            accumulant += vecA[d] * vecB[d]
        return (params["gamma"] * accumulant + params["coef0"]) ** params["degree"]

    def sigmoidSelf(self, vecA, params):
        accumulant = 0.0
        for d in range(params["dFeatures"]):
            accumulant += vecA[d] * vecA[d]
        return tanh(params["gamma"] * accumulant + params["coef0"])

    def sigmoid(self, vecA, vecB, params):
        accumulant = 0.0
        for d in range(params["dFeatures"]):
            accumulant += vecA[d] * vecB[d]
        return tanh(params["gamma"] * accumulant + params["coef0"])

#incomplete
class Controller(object):
    def __init__(self, initialGap, currentMethod, samplingInterval, problemSize):
        self.progress = [initialGap]
        self.method = []
        self.currentMethod = currentMethod
        if self.currentMethod == 3: #adaptive
            self.adaptive = True
            self.currentMethod = 1 #secondorder
        else:
            self.adaptive = False
        self.samplingInterval = samplingInterval
        self.inspectionPeriod = problemSize/(10.0*samplingInterval)

        self.timeSinceInspection = self.samplingInterval - 2
        self.beginningOfEpoch = 0
        self.rates = [0,0]
        self.currentInspectionPhase = 0

    def addIteration(self, gap):
        self.progress.append(gap)
        self.method.append(self.currentMethod)

    def findRate(self):
        pass
    def getMethod(self):
        if not self.adaptive:
            if self.currentMethod == 2: #random
                if random() < 0.5:
                    return 1 #second order
                else:
                    return 0 #first order
            else:
                return self.currentMethod
        #TODO: Adaptive Algorithm

if __name__ == '__main__':
    os.chdir('../examples')
    execfile('test.py')