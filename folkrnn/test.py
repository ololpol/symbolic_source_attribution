import theano
#import lasagne
import os
import sys
import time
import importlib
if sys.version_info < (3,0):
    import cPickle as pickle
else:
    import pickle
import numpy as np
#import theano.tensor as T
#from lasagne.layers import *