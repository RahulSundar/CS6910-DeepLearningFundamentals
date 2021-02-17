import numpy as np


def SquaredErrorLoss(X,Y,f):
    err = 0.0
    for x,y in zip(X,Y):
        fx = f(w,b,x)
        err += 0.5 * (fx - y)**2
    return err
    
def meanSquaredErrorLoss(X,Y,f):
    pass


def crossEntropyLoss():
    pass
