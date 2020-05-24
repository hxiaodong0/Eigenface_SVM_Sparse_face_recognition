###########################Initialization
import pickle
from scipy.io import loadmat
import time
import numpy as np
from numpy.linalg import norm
import random
#select numumber of training data and testing data for each person outof 38 person;
lst_m = [10, 20, 30, 40, 50]  # number of training image
m= lst_m[4]  # initialization of the nummber of training set
start = time.time()
import math
loadtemp = loadmat('YaleB_32x32.mat')
gnd = loadtemp["gnd"]  # 2414 X 1
gnd_copy = np.copy(gnd)
fea = loadtemp["fea"]  # (2414, 1024)
fea_copy2 = np.copy(fea)
n_training = []
n_testing = []

unique, counts = np.unique(gnd, return_counts=True)
D = dict(zip(unique, counts))
for i in range(len(counts)):
    n_training.append(m)  # random.choice(m)
for i in range(len(counts)):
    diff_temp = int(D[i+1]) - int(n_training[i])
    n_testing.append(diff_temp)

# select index in training
########################################################
def list_gen(n):
    lst = []
    for i in range(int(n)):
        lst.append(i + 1)
    return lst

def rand_index():
#index slicing:
    conn = []
    conn1 = []
    for Dic in D:
        card = list_gen(D[Dic])
        lst_training = []
        lst_testing = []
        for item in n_training:
            # temp1 = abs(int(Dic) - int(item))+1
            random.shuffle(card)
            lst_training.append(card[:int(item)])
            lst_testing.append(card[int(item):])
            break
        conn.append(lst_training)
        conn1.append(lst_testing)
    conn[0][0] = conn[0][0]
    conn[1][0] = [element + D[1] for element in conn[1][0]]
    conn[2][0] = [element + D[2] + D[1]  for element in conn[2][0]]
    conn[3][0] = [element + D[2] + D[1]+D[3] for element in conn[3][0]]
    conn[4][0] = [element + D[2] + D[1]+D[3]+D[4] for element in conn[4][0]]
    conn[5][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5] for element in conn[5][0]]
    conn[6][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6] for element in conn[6][0]]
    conn[7][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7] for element in conn[7][0]]
    conn[8][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8] for element in conn[8][0]]
    conn[9][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9] for element in conn[9][0]]
    conn[10][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10] for element in conn[10][0]]
    conn[11][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11] for element in conn[11][0]]
    conn[12][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12] for element in conn[12][0]]
    conn[13][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13] for element in conn[13][0]]
    conn[14][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14] for element in conn[14][0]]
    conn[15][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15] for element in conn[15][0]]
    conn[16][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16] for element in conn[16][0]]
    conn[17][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17] for element in conn[17][0]]
    conn[18][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18] for element in conn[18][0]]
    conn[19][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19] for element in conn[19][0]]
    conn[20][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20] for element in conn[20][0]]
    conn[21][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]for element in conn[21][0]]
    conn[22][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]+D[22] for element in conn[22][0]]
    conn[23][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]+D[22]+D[23] for element in conn[23][0]]
    conn[24][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]+D[22]+D[23]+D[24] for element in conn[24][0]]
    conn[25][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]+D[22]+D[23]+D[24]+D[25] for element in conn[25][0]]
    conn[26][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]+D[22]+D[23]+D[24]+D[25]+D[26] for element in conn[26][0]]
    conn[27][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]+D[22]+D[23]+D[24]+D[25]+D[26]+D[27] for element in conn[27][0]]
    conn[28][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]+D[22]+D[23]+D[24]+D[25]+D[26]+D[27]+D[28] for element in conn[28][0]]
    conn[29][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]+D[22]+D[23]+D[24]+D[25]+D[26]+D[27]+D[28]+D[29] for element in conn[29][0]]
    conn[30][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]+D[22]+D[23]+D[24]+D[25]+D[26]+D[27]+D[28]+D[29]+D[30] for element in conn[30][0]]
    conn[31][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]+D[22]+D[23]+D[24]+D[25]+D[26]+D[27]+D[28]+D[29]+D[30]+D[31] for element in conn[31][0]]
    conn[32][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]+D[22]+D[23]+D[24]+D[25]+D[26]+D[27]+D[28]+D[29]+D[30]+D[31]+D[32] for element in conn[32][0]]
    conn[33][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]+D[22]+D[23]+D[24]+D[25]+D[26]+D[27]+D[28]+D[29]+D[30]+D[31]+D[32]+D[33] for element in conn[33][0]]
    conn[34][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]+D[22]+D[23]+D[24]+D[25]+D[26]+D[27]+D[28]+D[29]+D[30]+D[31]+D[32]+D[33]+D[34] for element in conn[34][0]]
    conn[35][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]+D[22]+D[23]+D[24]+D[25]+D[26]+D[27]+D[28]+D[29]+D[30]+D[31]+D[32]+D[33]+D[34]+D[35] for element in conn[35][0]]
    conn[36][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]+D[22]+D[23]+D[24]+D[25]+D[26]+D[27]+D[28]+D[29]+D[30]+D[31]+D[32]+D[33]+D[34]+D[35]+D[36] for element in conn[36][0]]
    conn[37][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]+D[22]+D[23]+D[24]+D[25]+D[26]+D[27]+D[28]+D[29]+D[30]+D[31]+D[32]+D[33]+D[34]+D[35]+D[36]+D[37] for element in conn[37][0]]

    conn1[0][0] = conn1[0][0]
    conn1[1][0] = [element + D[1] for element in conn1[1][0]]
    conn1[2][0] = [element + D[2] + D[1]  for element in conn1[2][0]]
    conn1[3][0] = [element + D[2] + D[1]+D[3] for element in conn1[3][0]]
    conn1[4][0] = [element + D[2] + D[1]+D[3]+D[4] for element in conn1[4][0]]
    conn1[5][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5] for element in conn1[5][0]]
    conn1[6][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6] for element in conn1[6][0]]
    conn1[7][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7] for element in conn1[7][0]]
    conn1[8][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8] for element in conn1[8][0]]
    conn1[9][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9] for element in conn1[9][0]]
    conn1[10][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10] for element in conn1[10][0]]
    conn1[11][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11] for element in conn1[11][0]]
    conn1[12][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12] for element in conn1[12][0]]
    conn1[13][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13] for element in conn1[13][0]]
    conn1[14][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14] for element in conn1[14][0]]
    conn1[15][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15] for element in conn1[15][0]]
    conn1[16][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16] for element in conn1[16][0]]
    conn1[17][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17] for element in conn1[17][0]]
    conn1[18][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18] for element in conn1[18][0]]
    conn1[19][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19] for element in conn1[19][0]]
    conn1[20][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20] for element in conn1[20][0]]
    conn1[21][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]for element in conn1[21][0]]
    conn1[22][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]+D[22] for element in conn1[22][0]]
    conn1[23][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]+D[22]+D[23] for element in conn1[23][0]]
    conn1[24][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]+D[22]+D[23]+D[24] for element in conn1[24][0]]
    conn1[25][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]+D[22]+D[23]+D[24]+D[25] for element in conn1[25][0]]
    conn1[26][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]+D[22]+D[23]+D[24]+D[25]+D[26] for element in conn1[26][0]]
    conn1[27][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]+D[22]+D[23]+D[24]+D[25]+D[26]+D[27] for element in conn1[27][0]]
    conn1[28][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]+D[22]+D[23]+D[24]+D[25]+D[26]+D[27]+D[28] for element in conn1[28][0]]
    conn1[29][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]+D[22]+D[23]+D[24]+D[25]+D[26]+D[27]+D[28]+D[29] for element in conn1[29][0]]
    conn1[30][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]+D[22]+D[23]+D[24]+D[25]+D[26]+D[27]+D[28]+D[29]+D[30] for element in conn1[30][0]]
    conn1[31][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]+D[22]+D[23]+D[24]+D[25]+D[26]+D[27]+D[28]+D[29]+D[30]+D[31] for element in conn1[31][0]]
    conn1[32][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]+D[22]+D[23]+D[24]+D[25]+D[26]+D[27]+D[28]+D[29]+D[30]+D[31]+D[32] for element in conn1[32][0]]
    conn1[33][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]+D[22]+D[23]+D[24]+D[25]+D[26]+D[27]+D[28]+D[29]+D[30]+D[31]+D[32]+D[33] for element in conn1[33][0]]
    conn1[34][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]+D[22]+D[23]+D[24]+D[25]+D[26]+D[27]+D[28]+D[29]+D[30]+D[31]+D[32]+D[33]+D[34] for element in conn1[34][0]]
    conn1[35][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]+D[22]+D[23]+D[24]+D[25]+D[26]+D[27]+D[28]+D[29]+D[30]+D[31]+D[32]+D[33]+D[34]+D[35] for element in conn1[35][0]]
    conn1[36][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]+D[22]+D[23]+D[24]+D[25]+D[26]+D[27]+D[28]+D[29]+D[30]+D[31]+D[32]+D[33]+D[34]+D[35]+D[36] for element in conn1[36][0]]
    conn1[37][0] = [element + D[2] + D[1]+D[3]+D[4]+D[5]+D[6]+D[7]+D[8]+D[9]+D[10]+D[11]+D[12]+D[13]+D[14]+D[15]+D[16]+D[17]+D[18]+D[19]+D[20]+D[21]+D[22]+D[23]+D[24]+D[25]+D[26]+D[27]+D[28]+D[29]+D[30]+D[31]+D[32]+D[33]+D[34]+D[35]+D[36]+D[37] for element in conn1[37][0]]


    index_testing = []
    index_training = []
    for i in range(len(conn)):
        for itemt in (conn[i][0]):
            index_training.append(itemt)
    for i in range(len(conn1)):
        for itemx in (conn1[i][0]):
            index_testing.append(itemx)
    return conn, conn1, index_testing , index_training
#######training set



conn, conn1, index_testing, index_training = rand_index()

with open('m50.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([conn, conn1, index_testing, index_training], f)





