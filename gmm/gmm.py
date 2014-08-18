"""
specializer gmm-specializer
"""
import numpy as np
from numpy.random import random
from ctypes import *
from ctree.templates.nodes import *
from ctree.c.nodes import *
from ctree.jit import LazySpecializedFunction
from ctree.transformations import *
from ctree.jit import ConcreteSpecializedFunction
import unittest
from numpy.ctypeslib import as_array


class GMMComponents(object):
    """
    The Python interface to the components of a GMM.
    """
    def __init__(self, M, D, weights = None, means = None, covars = None):
        self.M = M
        self.D = D
        self.weights = weights if weights is not None else np.empty(M, dtype=np.float32)
        self.means = means if means is not None else  np.empty(M*D, dtype=np.float32)
        self.covars = covars if covars is not None else  np.empty(M*D*D, dtype=np.float32)
        self.comp_probs = np.empty(M, dtype=np.float32)

    def init_random_weights(self):
        self.weights = random((self.M))

    def init_random_means(self):
        self.means = random((self.M,self.D))

    def init_random_covars(self):
        self.covars = random((self.M, self.D, self.D))

    def shrink_components(self, new_M):
        self.weights = np.resize(self.weights, new_M)
        self.means = np.resize(self.means, new_M*self.D)
        self.covars = np.resize(self.covars, new_M*self.D*self.D)


class GMMEvalData(object):
    """
    The Python interface to the evaluation data generated by scoring a GMM.
    """
    def __init__(self, N, M):
        self.N = N
        self.M = M
        self.memberships = np.zeros((M,N), dtype=np.float32)
        self.loglikelihoods = np.zeros(N, dtype=np.float32)
        self.likelihood = 0.0

    def resize(self, N, M):
        self.memberships.resize((M,N))
        self.memberships = np.ascontiguousarray(self.memberships)
        self.loglikelihoods.resize(N, refcheck=False)
        self.loglikelihoods = np.ascontiguousarray(self.loglikelihoods)
        self.M = M
        self.N = N



class GMM(object):
    """
    The specialized GMM abstraction.
    """
    cvtype_name_list = ['diag','full']

    def __init__(self, M, D, means=None, covars=None, weights=None, cvtype='diag'):
        """
        cvtype must be one of 'diag' or 'full'. Uninitialized components will be seeded.
        """
        self.M = M
        self.D = D

        if cvtype in GMM.cvtype_name_list:
            self.cvtype = cvtype
        else:
            raise RuntimeError("Specified cvtype is not allowed, try one of " + str(GMM.cvtype_name_list))

        self.components = GMMComponents(M, D, weights, means, covars)
        self.eval_data = GMMEvalData(1, M)
        self.Ocltrain = Ocltrain(None)




    # ===== Specializer functions used by the app writers =====

    # Getters

    def get_all_component_weights(self):
        return self.components.weights

    def get_one_component_weights(self, component_id):
        return self.components.weights[component_id]

    def get_all_component_means(self):
        return self.components.means

    def get_one_component_means(self, component_id):
        return self.components.means[component_id]

    def get_all_component_full_covariance(self):
        return self.components.covars

    def get_one_component_full_covariance(self, component_id):
        return self.components.covars[component_id]

    def get_all_component_diag_covariance(self):
        full_covar = self.components.covars
        diags = []
        for m in range(self.M):
            diags.append(np.diag(full_covar[m]))
        return np.concatenate(diags)

    def get_one_component_diag_covariance(self, component_id):
        full_covar = self.components.covars[component_id]
        return np.diag(full_covar)

    # Training and Evaluation of GMM

    def train_using_python(self, input_data, iters=10):
        from sklearn import mixture
        self.clf = mixture.GMM(n_components=self.M, covariance_type=self.cvtype)
        self.clf.fit(input_data)
        return self.clf.means_, self.clf.covars_

    def eval_using_python(self, obs_data):
        if self.clf is not None:
            return self.clf.eval(obs_data)
        else: return []

    def predict_using_python(self, obs_data):
        if self.clf is not None:
            return self.clf.predict(obs_data)
        else: return []



    def train(self, input_data, min_em_iters=1, max_em_iters=10):
        """
        Train the GMM on the data. Optinally specify max and min iterations.
        Not completed
        """
        N = input_data.shape[0]
        if input_data.shape[1] != self.D:
            print "Error: Data has %d features, model expects %d features." % (input_data.shape[1], self.D)

        self.eval_data.likelihood,self.components.means,self.components.covars = self.Ocltrain(input_data,self.eval_data.memberships,self.eval_data.loglikelihoods,self.M,self.D,N,min_em_iters,max_em_iters,self.cvtype,self.eval_data.likelihood)
        self.components.means = self.components.means.reshape(self.M, self.D)
        self.components.covars = self.components.covars.reshape(self.M, self.D, self.D)

        return self.eval_data.likelihood


    def eval(self, obs_data):
        N = obs_data.shape[0]
        if obs_data.shape[1] != self.D:
            print "Error: Data has %d features, model expects %d features." % (obs_data.shape[1], self.D)

        logprob = self.eval_data.loglikelihoods
        posteriors = self.eval_data.memberships
        return logprob, posteriors # N log probabilities, NxM posterior probabilities for each component

    def score(self, obs_data):
        logprob, posteriors = self.eval(obs_data)
        return logprob # N log probabilities

    def decode(self, obs_data):
        logprob, posteriors = self.eval(obs_data)
        return logprob, posteriors.argmax(axis=0) # N log probabilities, N indexes of most likely components

    def predict(self, obs_data):
        logprob, posteriors = self.eval(obs_data)
        return posteriors.argmax(axis=0) # N indexes of most likely components

class Ocltrain(LazySpecializedFunction):


    def args_to_subconfig(self, args):
        return (args[0])


    def transform(self, tree, program_config):

        kernelFunc = program_config[0]
        kernelPath = os.path.join(os.getcwd(), "..", "templates","training_kernel.c")
        kernelInserts = {
            "kernelFunc": SymbolRef(kernelFunc),
        }
        kernel = CFile("em_train", [
            FileTemplate(kernelPath, kernelInserts)
        ])

        entry_type = CFUNCTYPE(c_int,POINTER(c_float), POINTER(c_float), POINTER(c_float),
                               c_int, c_int, c_int,
                               c_int, c_int,POINTER(c_char),
                               POINTER(c_float),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)))


        proj = Project([kernel])

        fn = CtrainFunction()

        return fn.finalize('em_train', proj,entry_type)

class CtrainFunction(ConcreteSpecializedFunction):
    def finalize(self, entry_point_name, project_node, entry_typesig):
        self._c_function = self._compile(entry_point_name, project_node, entry_typesig)

        return self

    def __call__(self, *args):
        input_data,component_memberships,loglikelihoods,num_components,num_dimensions,num_events,min_iters, max_iters,cvtype, ret_likelihood = args
        #print input_data
        input_data =input_data.ctypes.data_as(POINTER(c_float))
        component_memberships = component_memberships.ctypes.data_as(POINTER(c_float))
        loglikelihoods = loglikelihoods.ctypes.data_as(POINTER(c_float))

        #return value
        ret_likelihood = c_float()
        ret_means = pointer(c_float())
        ret_covar = pointer(c_float())
        self._c_function(input_data,component_memberships,loglikelihoods,num_components,num_dimensions,num_events,min_iters, max_iters,cvtype, byref(ret_likelihood),byref(ret_means),byref(ret_covar))

        return ret_likelihood.value,as_array(ret_means,shape=(num_components* num_dimensions,)),as_array(ret_covar,shape=(num_components* num_dimensions* num_dimensions,))

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start


class GmmTest(unittest.TestCase):
    def setUp(self):
        self.D = 2
        self.N = 10
        self.M = 3
        np.random.seed(0)
        C = np.array([[0., -0.7], [3.5, .7]])
        C1 = np.array([[-0.4, 1.7], [0.3, .7]])
        Y = np.r_[
            np.dot(np.random.randn(self.N/3, 2), C1),
            np.dot(np.random.randn(self.N/3, 2), C),
            np.random.randn(self.N/3, 2) + np.array([3, 3]),
            ]
        self.X = Y.astype(np.float32)

    def test_pure_python(self):
        gmm = GMM(self.M, self.D, cvtype='diag')
        means, covars = gmm.train_using_python(self.X)
        print "training data"
        print self.X
        print "pure result means:"
        print means
        print "pure result covars:"
        print covars
        Y = gmm.predict_using_python(self.X)
        self.assertTrue(len(set(Y)) > 1)

    def test_training_once(self):
        print "test training once"
        gmm0 = GMM(self.M, self.D, cvtype='diag')
        likelihood0 = gmm0.train(self.X)

        means0  = gmm0.components.means
        covars0 = gmm0.components.covars
        print "likelihood0"
        print likelihood0
        print "means0"
        print means0
        print "covars0"
        print covars0

        gmm1 = GMM(self.M, self.D, cvtype='diag')
        likelihood1 = gmm1.train(self.X)
        print "likelihood1"
        print likelihood1
        means1  = gmm1.components.means
        print "means1"
        print means1
        covars1 = gmm1.components.covars
        print "covars1"
        print covars1

        self.assertAlmostEqual(likelihood0, likelihood1, places=3)
        for a,b in zip(means0.flatten(), means1.flatten()): self.assertAlmostEqual(a,b, places=3)
        for a,b in zip(covars0.flatten(), covars1.flatten()): self.assertAlmostEqual(a,b, places=3)
        print "fin"





def main():
    D = 2
    N = 600
    M = 3
    np.random.seed(0)
    C = np.array([[0., -0.7], [3.5, .7]])
    C1 = np.array([[-0.4, 1.7], [0.3, .7]])
    Y = np.r_[np.dot(np.random.randn(N/3, 2), C1),np.dot(np.random.randn(N/3, 2), C),np.random.randn(N/3, 2) + np.array([3, 3]),]
    X = Y.astype(np.float32)


    gmm = GMM(M, D, cvtype='diag')
    means, covars = gmm.train_using_python(X)
    Y = gmm.predict_using_python(X)
    np.testing.assertTrue(len(set(Y)) > 1)




    gmm0 = GMM(M, D, cvtype='diag')
    likelihood0 = gmm0.train(X)
    means0  = gmm0.components.means.flatten()
    covars0 = gmm0.components.covars.flatten()


    gmm1 = GMM(M, D, cvtype='diag')
    likelihood1 = gmm1.train(X)
    means1  = gmm1.components.means.flatten()
    covars1 = gmm1.components.covars.flatten()

    self.assertAlmostEqual(likelihood0, likelihood1, places=3)
    for a,b in zip(means0, means1):   np.testing.assertAlmostEqual(a,b, places=3)
    for a,b in zip(covars0, covars1): np.testing.assertAlmostEqual(a,b, places=3)



if __name__ == '__main__':

    unittest.main()

