from load import *
from datasegmentation import D, unique, counts
import numpy as np
from sklearn import lda
# from sklearn import datasets
# digits = datasets.load_digits()
def svmf(index_train , index_test):
    cnt = 0
    err = 0
    clf = lda.LDA(gamma = 'scale')
    X = fea_copy[index_train]
    y = gnd[index_train]
    clf.fit(X, y.ravel())
    for i in range(len(index_test)):
        if clf.predict(fea_copy[[index_test[i]]]) != gnd[index_test[i]]:
            err += 1
        cnt += 1
    return cnt,err
cnt10, err10 = svmf(m10index_training,m10index_testing)
cnt20, err20 = svmf(m20index_training,m20index_testing)
cnt30, err30 = svmf(m30index_training,m40index_testing)
cnt40, err40 = svmf(m40index_training,m40index_testing)
cnt50, err50 = svmf(m50index_training,m50index_testing)

eigenfaces_error_rate = pandas.DataFrame(np.array([cnt10, err10, err10/cnt10*100]),columns = ["m10"])
eigenfaces_error_rate['m20'] = np.array([cnt20, err20, err20 / cnt20 * 100])
eigenfaces_error_rate['m30'] = np.array([cnt30, err30, err30 / cnt30 * 100])
eigenfaces_error_rate['m40'] = np.array([cnt40, err40, err40 / cnt40 * 100])
eigenfaces_error_rate['m50'] = np.array([cnt50, err50, err50 / cnt50 * 100])

print(eigenfaces_error_rate)

def plot():
    output_file("line.html", title="SVM vs n_of_training_samples")
    p = figure(plot_width=1000, plot_height=400)
    p.xaxis.axis_label = "number of training samples"
    p.yaxis.axis_label = "Error rate %"
    # add a line renderer
    p.circle([10,20,30,40,50],eigenfaces_error_rate.iloc[2],line_width=2)
    show(p)
    return False

