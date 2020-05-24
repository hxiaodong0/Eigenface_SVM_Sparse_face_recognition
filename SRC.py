from load import *
import sklearn.decomposition
from datasegmentation import D, unique, counts
from sklearn import preprocessing
# from spicy import sparse

np.seterr(divide='ignore', invalid='ignore')
yi = unique   #{y1,....,yc} c classes
xi = counts # N_training images per class

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
ui = np.array(ui)
def D_cal_each_class():
    n = 0
    lstc = []
    global c1
    c1 = fea_copy[:xi[0]]
    lstc.append(c1)
    for i in range(len(xi)-1):   #    for i in range():
        globals()['c%d' % int(i+2)] = fea_copy[xi[i]+n : xi[i]+n+xi[i+1]]
        n += xi[i]
        lstc.append(eval('c%d' % int(i + 1)))
    lst = []
    return lstc
kk = D_cal_each_class()
#training
def DD(lst = kk, index = m10conn):
    X = kk
    D = sklearn.decomposition.MiniBatchDictionaryLearning(n_components=1, alpha=1, n_iter=100)
    y = index
    lstc = []
    for i in range(1,len(X)+1):   #    for i in range():
        locals()['d%d' % int(i)] = D.fit(eval('c%d' % int(i))).components_
        lstc.append(eval('d%d' % int(i)))
    return lstc

DD = DD(kk, m10conn)
DD = np.array(DD)
DD = np.reshape(DD,(38,1024))
y = np.dot(DD,fea_copy.T)
# y = y.reshape(y,(38,1024))
#testing
def src(index_train = DD , index_test = m10index_testing, index_training = m10training ):
    DD = index_train
    testing0 = np.array(())
    cnt = 0
    err = 1
    temp = list()
    temp1 = list()

    for i in range(len(index_test)):
        l = np.dot(DD, fea_copy[index_test[i]]) #, true_index,predict_index
        temp.append(l)
    temp = np.array(temp)

    for i in range(len(temp)):
        for j in range(len(DD)):
            f = np.reshape(temp[i],(38,1))
            k = np.reshape(DD[j],(1,1024))
            dist = f * k
            norm = np.linalg.norm(dist)
            temp1.append(norm)
        p = (np.min(temp1, axis=0))  # p returns the minimum distance
        indexp = np.argwhere(temp1 == p)
        if len(indexp) != 1:
            indexp = indexp[0]
        indexp = int(indexp/38)
        cnt += 1
        if [gnd[index_test[i]]] != [gnd[index_test[indexp]]]:
            err += 1

    for i in range(len(index_test)):
        img = fea_copy[index_test[i]]
        true_index = gnd[index_test[i]] # 2034 test data, 2034 indexs
        err/=index_training
        break
    for i in range(1,len(fea_copy[m10conn[1]])):
        break
    return cnt,err

cnt10,err10 = src(DD, m10index_testing,m10training)
cnt20,err20 = src(DD, m20index_testing,m20training)
cnt30,err30 = src(DD, m30index_testing,m30training)
cnt40,err40 = src(DD, m40index_testing,m40training)
cnt50,err50 = src(DD, m50index_testing,m50training)
eigenfaces_error_rate = pandas.DataFrame(np.array([cnt10, err10, err10 / cnt10 * 100]), columns=["m10"])
eigenfaces_error_rate['m20'] = np.array([cnt20, err20, err20 / cnt20 * 100])
eigenfaces_error_rate['m30'] = np.array([cnt30, err30, err30 / cnt30 * 100])
eigenfaces_error_rate['m40'] = np.array([cnt40, err40, err40 / cnt40 * 100])
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
plot()
print(eigenfaces_error_rate)
#Save data
# with open('eigen_faces_0_50', 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump([SRC_error_rate], f)