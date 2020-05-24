from load import *
from datasegmentation import D, unique, counts
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
yi = unique   #{y1,....,yc} c classes
xi = counts # N_training images per class
from scipy.linalg import norm

def class_index(xi=xi,yi=yi):
    n = 0
    lstc = []
    c1 = fea_copy[:xi[0]]
    lstc.append(c1)
    for i in range(len(xi)-1):   #    for i in range():
        locals()['c%d' % int(i+2)] = fea_copy[xi[i]+n : xi[i]+n+xi[i+1]]
        n += xi[i]
        lstc.append(eval('c%d' % int(i + 1)))
    lst = []
    for i in range(len(xi)):
        locals()['u%d' % int(i+1)] = np.mean(eval('c%d' % int(i+1)), axis = 0)
        lst.append(eval('u%d' % int(i + 1)))
    return lst,lstc
ui, ci = class_index(xi,yi) # ui = average of each class; ci = index of Total average
def u(ui= ui):
    u = np.mean(ui, axis= 0)
    return u
u = np.array(u(ui))
# scatter calculation
# 1 scatter of classi

def s_w(ui = ui):      #within class scatter
    diff = []
    lst = []
    sum = np.zeros(shape=(1024,1024))
    for i in range(len(ui)):
        ui[i] = np.transpose(ui[i])
        diff = (ci[i]-np.array([ui[0],]*len(ci[i])))
        diff_T = np.transpose(diff)
        si = np.dot(diff_T,diff)
        locals()['Si%d' % int(i + 1)] = si
        sum += si
    # np.sum()
    return sum
swsum = s_w(ui)
# swsum = swsum / np.linalg.norm(swsum)
  # within class scatter for each cluster i  64*64 large numbers
def s_B(ui= ui,u=u):       #between class scatter
    diff = []
    lst = []
    sum = np.zeros(shape=(1024,1024))
    lst = []
    for i in range(len(yi)):
        diff = (ui[i] - np.array([u[0], ] * len(ui[i])))
        diff = diff.reshape(1, len(diff))
        diff_T = np.transpose(diff)
        diff_T = diff_T.reshape(len(diff_T), 1)
        sbi = np.abs(yi[i]) * np.multiply(diff, diff_T)
        locals()['Sbi%d' % int(i + 1)] = sbi
    sum += sbi
    return sum
sbsum = s_B(ui,u)
# sbsum = sbsum / np.linalg.norm(sbsum)

def SVD_calculation(swsum=swsum ,sbsum = sbsum, n_pc=50,discard_first_n_vectors = 3):
    U1,S1,V1 = np.linalg.svd(swsum)
    U2,S2,V2 = np.linalg.svd(sbsum)
    components1 = V1[discard_first_n_vectors:n_pc]
    projected1  = U1[:,:n_pc]*S1[:n_pc]
    components2 = V2[discard_first_n_vectors:n_pc]
    projected2  = U2[:,:n_pc]*S2[:n_pc]
    return components1, projected1 ,components2, projected2

sb, sb_projected1 ,sw, sw_projected2 = SVD_calculation(swsum , sbsum, 50, 3)

def maximization(sb= sb,sw=sw):
    max_sb = np.argmax(sb, axis=0)
    min_sw = np.argmin(sw, axis=0)
    wopt = np.argmax(sb/sw, axis=0)
    return max_sb,min_sw, wopt

max_sb,min_sw, C = maximization(sb,sw)
# C = max_sb*min_sw
# C = max_sb/min_sw

def training_set(X):
    n_components = 50
    d = []
    face_index = np.array((gnd_copy[0]))
    w = C * (fea_copy[X[0]] - u)
    for i in range(1,len(fea_copy[X])):
        w = np.vstack((w, (C * (fea_copy[X[i]] - u))))
        # w = np.dot(C, (fea_copy[m10index_training[i]] - M))
        face_index = np.vstack((face_index, gnd_copy[X[i]]))
    return w, face_index

def testing_set(X):
    n_components = 50
    face_index = np.array((gnd_copy[0]))
    w = C * (fea_copy[X[0]] - u)
    for i in range(1,len(fea_copy[X])):
        w = np.vstack((w, (C * (fea_copy[X[i]] - u))))
        # w = np.dot(C, (fea_copy[m10index_training[i]] - M))
        face_index = np.vstack((face_index , gnd_copy[X[i]]))
    return w, face_index


def compare(w_test, index_test, w_train, index_train):
    cnt = 0
    err = 0
    for i in range(len(w_test)):
        temp = []
        for j in range(len(w_train)):
            dist = np.linalg.norm(w_test[i] - w_train[j])
            temp.append((dist,index_test[i][0],index_train[j][0]))
        try:
            p = (np.min(temp,axis=0))[0]   # p returns the minimum distance
            indexp = np.argwhere(temp == p)
            cnt += 1
            if temp[indexp[0][0]][1] != temp[indexp[0][0]][2]:
                err +=1
        except:
            pass
    return cnt,err


w_training10, face_index_training10  = training_set(m10index_training)
w_testing10, face_index_testing10  = testing_set(m10index_testing)
if np.isnan(w_training10).any() == True:
    w_training10 = np.nan_to_num(w_training10)
if  np.isnan(face_index_training10).any() == True:
    face_index_training10 = np.nan_to_num(face_index_training10)
if  np.isnan(w_testing10).any() == True:
    w_testing10 = np.nan_to_num(w_testing10)
if  np.isnan(face_index_testing10).any() == True:
    face_index_testing10 = np.nan_to_num(face_index_testing10)
cnt10,err10 = compare(w_testing10,face_index_testing10,w_training10,face_index_training10)
eigenfaces_error_rate = pandas.DataFrame(np.array([cnt10, err10, err10/cnt10*100]),columns = ["m10"])


w_training20, face_index_training20 = training_set(m20index_training)
w_testing20, face_index_testing20 = testing_set(m20index_testing)
if np.isnan(w_training20).any() == True:
    w_training20 = np.nan_to_num(w_training20)
if  np.isnan(face_index_training20).any() == True:
    face_index_training20 = np.nan_to_num(face_index_training20)
if  np.isnan(w_testing20).any() == True:
    w_testing20 = np.nan_to_num(w_testing20)
if  np.isnan(face_index_testing20).any() == True:
    face_index_testing20 = np.nan_to_num(face_index_testing20)
cnt20, err20 = compare(w_testing20, face_index_testing20, w_training20, face_index_training20)
eigenfaces_error_rate['m20'] = np.array([cnt20, err20, err20 / cnt20 * 100])

w_training30, face_index_training30 = training_set(m30index_training)
w_testing30, face_index_testing30 = testing_set(m30index_testing)
if np.isnan(w_training30).any() == True:
    w_training30 = np.nan_to_num(w_training30)
if  np.isnan(face_index_training30).any() == True:
    face_index_training30 = np.nan_to_num(face_index_training30)
if  np.isnan(w_testing30).any() == True:
    w_testing30 = np.nan_to_num(w_testing30)
if  np.isnan(face_index_testing30).any() == True:
    face_index_testing30 = np.nan_to_num(face_index_testing30)
cnt30, err30 = compare(w_testing30, face_index_testing30, w_training30, face_index_training30)
eigenfaces_error_rate['m30'] = np.array([cnt30, err30, err30 / cnt30 * 100])

w_training40, face_index_training40 = training_set(m40index_training)
w_testing40, face_index_testing40 = testing_set(m40index_testing)
if np.isnan(w_training40).any() == True:
    w_training40 = np.nan_to_num(w_training40)
if  np.isnan(face_index_training40).any() == True:
    face_index_training40 = np.nan_to_num(face_index_training40)
if  np.isnan(w_testing40).any() == True:
    w_testing40 = np.nan_to_num(w_testing40)
if  np.isnan(face_index_testing40).any() == True:
    face_index_testing40 = np.nan_to_num(face_index_testing40)
cnt40, err40 = compare(w_testing40, face_index_testing40, w_training40, face_index_training40)
eigenfaces_error_rate['m40'] = np.array([cnt40, err40, err40 / cnt40 * 100])

w_training50, face_index_training50 = training_set(m50index_training)
w_testing50, face_index_testing50 = testing_set(m50index_testing)
if np.isnan(w_training50).any() == True:
    w_training50 = np.nan_to_num(w_training50)
if  np.isnan(face_index_training50).any() == True:
    face_index_training50 = np.nan_to_num(face_index_training50)
if  np.isnan(w_testing50).any() == True:
    w_testing50 = np.nan_to_num(w_testing50)
if  np.isnan(face_index_testing50).any() == True:
    face_index_testing50 = np.nan_to_num(face_index_testing50)
cnt50, err50 = compare(w_testing50, face_index_testing50, w_training50, face_index_training50)
eigenfaces_error_rate['m50'] = np.array([cnt50, err50, err50 / cnt50 * 100])


def plot():
    output_file("line.html", title="question 1 k=1, error rate vs number of training sets")
    p = figure(plot_width=1000, plot_height=400)
    p.xaxis.axis_label = "number of training samples"
    p.yaxis.axis_label = "Error rate %"
    # add a line renderer
    p.circle([10,20,30,40,50],eigenfaces_error_rate.iloc[2],line_width=2)
    show(p)
    return False
print(eigenfaces_error_rate)
plot()
