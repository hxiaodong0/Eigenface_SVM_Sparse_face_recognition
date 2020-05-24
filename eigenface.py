# Eigenfaces, Fisherfaces, Support Vector Machine (SVM), and Sparse Representation-based Classification
# error function: number of errors/number of iamges *100%
from load import *

#Eigenfaces eigenfaces u1,u2,...uk that span that subspace ;representa all face images in the dataset as a linear combination of the eignfaces(dotproduct);
#Training
#compute average face x = 1/N sum(xi) for all training images
average_face =  np.sum(a = fea_copy, axis = 0)  * (1/len(gnd))
def show_face(n = average_face):
    img = n.reshape((32, 32))
    imgplot = plt.imshow(img, cmap=plt.cm.gray)
    plt.show()
def show_eigen(n = average_face):
    x  = np.resize(n,2401*1)
    img = x.reshape((49, 49))
    imgplot = plt.imshow(img, cmap=plt.cm.gray)
    plt.show()
#compute the difference image sigma(i) = x(i) - x_head
# pca = decomposition.PCA(n_components=150, whiten=True)
# pca.fit(m10index_training)
def plot_portraits(images, titles, h, w, n_row, n_col):
    plt.figure(figsize=(2.2 * n_col, 2.2 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.20)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())

def training():
    x = np.vstack([average_face] * len(fea_copy[m10index_training]))
    fea_diff = np.subtract(fea_copy[m10index_training],x)
    sigma_i = fea_diff
    sigma_i_transpose = np.transpose(sigma_i)
    #compute the total scatter amtrix ST W = vi = Bui
    st = np.dot(sigma_i, sigma_i_transpose)* (1/len(gnd))
    w, v = LA.eig(st)
    normv = v / np.linalg.norm(v, axis=0)
    zipped = dict(zip(w, normv))
    w = []
    for i in range((40)):
        max_eigen = sorted(zipped.keys())[:40]
        w.append(zipped[max_eigen[i]])
        zipped.pop(max_eigen[i])
    w_T = np.array(w)
    w_T = w_T.real
    w = np.transpose(w)
    w = w.real
    return fea_diff, w, w_T

def pca(X,n_pc=40,discard_first_n_vectors = 1):
    average_face = np.mean(X, axis=0)
    diff = X - average_face
    U,S,V = np.linalg.svd(diff)
    components = V[discard_first_n_vectors:n_pc]
    projected  = U[:,:n_pc]*S[:n_pc]
    return projected, components, average_face, diff
n_components = 100

P, C, M, Y = pca(fea_copy, n_components)

def training_set(X):
    d = []
    face_index = np.array((gnd_copy[0]))
    w = np.dot(C, (fea_copy[X[0]] - M))
    for i in range(1,len(fea_copy[X])):
        w = np.vstack((w, np.dot(C, (fea_copy[X[i]] - M))))
        # w = np.dot(C, (fea_copy[m10index_training[i]] - M))
        face_index = np.vstack((face_index , gnd_copy[X[i]]))
    return w, face_index

def testing_set(X):
    face_index = np.array((gnd_copy[0]))
    w = np.dot(C, (fea_copy[X[0]] - M))
    for i in range(1,len(fea_copy[X])):
        w = np.vstack((w, np.dot(C, (fea_copy[X[i]] - M))))
        # w = np.dot(C, (fea_copy[m10index_training[i]] - M))
        face_index = np.vstack((face_index , gnd_copy[X[i]]))
    return w, face_index

def show_all():
    show_face(C[0])
    eigenfaces = C.reshape((n_components, 32, 32))
    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    plot_portraits(eigenfaces, eigenface_titles, 32, 32, 4, 4)
    return 0

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
        # indexp = np.argwhere(temp == p)
        # cnt += 1
        # if temp[indexp[0][0]][1] != temp[indexp[0][0]][2]:
        #     err +=1
    return cnt,err

def eigenface_error_rate():

    w_training10, face_index_training10  = training_set(m10index_training)
    w_testing10, face_index_testing10  = testing_set(m10index_testing)
    cnt10,err10 = compare(w_testing10,face_index_testing10,w_training10,face_index_training10)
    eigenfaces_error_rate = pandas.DataFrame(np.array([cnt10, err10, err10/cnt10*100]),columns = ["m10"])

    w_training20, face_index_training20  = training_set(m20index_training)
    w_testing20, face_index_testing20  = testing_set(m20index_testing)
    cnt20,err20 = compare(w_testing20,face_index_testing20,w_training20,face_index_training20)
    eigenfaces_error_rate['m20'] = np.array([cnt20, err20, err20/cnt20*100])

    w_training30, face_index_training30  = training_set(m30index_training)
    w_testing30, face_index_testing30  = testing_set(m30index_testing)
    cnt30,err30 = compare(w_testing30,face_index_testing30,w_training30,face_index_training30)
    eigenfaces_error_rate['m30'] = np.array([cnt30, err30, err30/cnt30*100])

    w_training40, face_index_training40  = training_set(m40index_training)
    w_testing40, face_index_testing40  = testing_set(m40index_testing)
    cnt40,err40 = compare(w_testing40,face_index_testing40,w_training40,face_index_training40)
    eigenfaces_error_rate['m40'] = np.array([cnt40, err40, err40/cnt40*100])

    w_training50, face_index_training50  = training_set(m50index_training)
    w_testing50, face_index_testing50  = testing_set(m50index_testing)
    cnt50,err50 = compare(w_testing50,face_index_testing50,w_training50,face_index_training50)
    eigenfaces_error_rate['m50'] = np.array([cnt50, err50, err50/cnt50*100])
    return eigenfaces_error_rate

eigenfaces_error_rate = eigenface_error_rate()


def plot():
    output_file("line.html", title="question 1 k=1, error rate vs number of training sets")
    p = figure(plot_width=1000, plot_height=400)
    p.xaxis.axis_label = "number of training samples"
    p.yaxis.axis_label = "Error rate %"
    # add a line renderer
    p.circle([10,20,30,40,50],eigenfaces_error_rate.iloc[2],line_width=2)
    show(p)
    return False
# with open('eigen_faces_0_50', 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump([eigenfaces_error_rate], f)

plot()
print(print(eigenfaces_error_rate))