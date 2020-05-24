import pickle
import matplotlib.pyplot as plt

with open('eigen_faces_0_50','rb') as f:  # Python 3: open(..., 'rb')
    f = pickle.load(f)
with open('eigen_faces_3_50','rb') as f:  # Python 3: open(..., 'rb')
    f = pickle.load(f)
with open('LDA_3_50','rb') as f:  # Python 3: open(..., 'rb')
    f = pickle.load(f)
print(f)
