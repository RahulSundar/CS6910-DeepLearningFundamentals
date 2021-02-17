import numpy as np


def grad_w(w,b,x,y):
    fx = f(w,b,x)
    return (fx - y) * fx *(1-fx)*x
    
def grad_b(w,b,x,y):
    fx = f(w,b,x)
    return (fx - y) * fx *(1-fx)


def grad_activation():
    pass
