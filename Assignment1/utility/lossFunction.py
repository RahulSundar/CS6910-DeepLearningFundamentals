import numpy as np


def meanSquaredErrorLoss(Y_train,Y_pred):
    MSE = np.mean((Y_train - Y_pred)**2)
    return MSE
    
def crossEntropyLoss(P, Q):
    CE = [-P[i]*np.log(Q[i]) for i in range(len(Q)) ]
    crossEntropy = np.sum(CE)
    return crossEntropy
