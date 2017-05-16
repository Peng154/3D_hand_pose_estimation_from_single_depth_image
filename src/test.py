import numpy as np

L = np.matrix([[0, 0, 0],
               [-1, 0, 0],
               [0.5,0.5,0]])
U = np.matrix([[0, 0.5, -0.5],
               [0, 0, -1],
               [0, 0, 0]])
I = np.matrix([[1.,0.,0.],
              [0.,1.,0.],
              [0.,0.,1.]])
print(I-L)


print(np.linalg.eigvals((I-L).I.dot(U)))