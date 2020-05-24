import pickle
from scipy.io import loadmat
import numpy as np
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from sklearn.model_selection import cross_val_score
from numpy import linalg as LA
import pandas
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show,output_file

loadtemp = loadmat('YaleB_32x32.mat')
gnd = loadtemp["gnd"]  # 2414 X 1
gnd_copy = np.copy(gnd)
fea = loadtemp["fea"]  # (2414, 1024)
fea_copy = np.copy(fea)
n_components = 50
p = [2,3,4,4.5,5]
p2 =[10,20,30,40,50]
# import index for training and testing data from data
with open('m10.pkl','rb') as f:  # Python 3: open(..., 'rb')
    m10conn, m10conn1, m10index_testing, m10index_training = pickle.load(f)
m10index_testing = np.subtract(m10index_testing, 1)
m10index_training = np.subtract(m10index_training, 1)
with open('m20.pkl','rb') as f:  # Python 3: open(..., 'rb')
    m20conn, m20conn1, m20index_testing, m20index_training = pickle.load(f)
m20index_testing = np.subtract(m20index_testing, 1)
m20index_training = np.subtract(m20index_training, 1)
with open('m30.pkl','rb') as f:  # Python 3: open(..., 'rb')
    m30conn, m30conn1, m30index_testing, m30index_training = pickle.load(f)
m30index_testing = np.subtract(m30index_testing, 1)
m30index_training = np.subtract(m30index_training, 1)
with open('m40.pkl','rb') as f:  # Python 3: open(..., 'rb')
    m40conn, m40conn1, m40index_testing, m40index_training = pickle.load(f)
m40index_testing = np.subtract(m40index_testing, 1)
m40index_training = np.subtract(m40index_training, 1)
with open('m50.pkl','rb') as f:  # Python 3: open(..., 'rb')
    m50conn, m50conn1, m50index_testing, m50index_training = pickle.load(f)
m50index_testing = np.subtract(m50index_testing, 1)
m50index_training = np.subtract(m50index_training, 1)

for i in range(5):
    locals()['m%dtraining' % int(p2[i])] = p[i]