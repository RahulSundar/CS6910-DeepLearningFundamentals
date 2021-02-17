import numpy as np
init_w, init_b = -2.0 , -2.0
X = [2.5, 0.9]
Y = [0.5, 0.1]

from lossFunction import *
from activations import *
from gradientCalculations import *

class Optimiser():

    def __init__(self, max_epochs) 
        self.max_epochs = max_epochs
        
    
    def do_gradient_descent(self):

        w, b , eta = init_w, init_b, 1.0
        for i in range(self.max_epochs):
            dw, db  = 0.0, 0.0
            for x,y in zip(X,Y):
                dw += grad_w(w,b,x,y)
                db += grad_b(w,b,x,y)
            w = w - eta*dw
            b = b - eta*db
        return w, b


    def do_momentum_gradient_descent(self):

        w, b, eta = init_w, init_b, 1.0
        prev_v_w, prev_v_b, gamma = 0.0, 0.0, 0.9
        for i in range(self.max_epochs):
            dw, db  = 0.0, 0.0
            for x,y in zip(X,Y):
                dw += grad_w(w,b,x,y)
                db += grad_b(w,b,x,y)
                
            v_w = gamma*prev_v_w + eta*dw
            v_b = gamma*prev_v_b + eta*db
            
            w = w - v_w
            b = b - v_b
            
            prev_v_w = v_w
            prev_v_b = v_b
            
        return w, b
        
        
        
    def do_stochastic_momentum_gradient_descent(self):

        w, b, eta = init_w, init_b, 1.0
        prev_v_w, prev_v_b, gamma = 0.0, 0.0, 0.9
        for i in range(self.max_epochs):
            dw, db  = 0.0, 0.0
            for x,y in zip(X,Y):
                dw += grad_w(w,b,x,y)
                db += grad_b(w,b,x,y)
                
                v_w = gamma*prev_v_w + eta*dw
                v_b = gamma*prev_v_b + eta*db
                
                w = w - v_w
                b = b - v_b
                
                prev_v_w = v_w
                prev_v_b = v_b
            
        return w, b
            
    def do_nesterov_gradient_descent(self):

        w, b, eta = init_w, init_b, 1.0
        prev_v_w, prev_v_b, gamma = 0.0, 0.0, 0.9
        
        
        for i in range(self.max_epochs):
            dw, db  = 0.0, 0.0
            winter = w - gamma*prev_v_w
            binter = b - gamma*prev_v_b
            for x,y in zip(X,Y):
                dw += grad_w(winter,b,x,y)
                db += grad_b(winter,b,x,y)
                
            v_w = gamma*prev_v_w + eta*dw
            v_b = gamma*prev_v_b + eta*db
            
            w = w - v_w
            b = b - v_b
            
            prev_v_w = v_w
            prev_v_b = v_b
            
        return w, b        
            

    def do_stochastic_nesterov_gradient_descent(self):

        w, b, eta = init_w, init_b, 1.0
        prev_v_w, prev_v_b, gamma = 0.0, 0.0, 0.9
        
        
        for i in range(self.max_epochs):
            dw, db  = 0.0, 0.0
            winter = w - gamma*prev_v_w
            binter = b - gamma*prev_v_b
            for x,y in zip(X,Y):
                dw += grad_w(winter,b,x,y)
                db += grad_b(winter,b,x,y)
                
                v_w = gamma*prev_v_w + eta*dw
                v_b = gamma*prev_v_b + eta*db
                
                w = w - v_w
                b = b - v_b
                
                prev_v_w = v_w
                prev_v_b = v_b
            
        return w, b 
            
            
    def do_stochastic_gradient_descent(self):

        w, b , eta = init_w, init_b, 1.0
        for i in range(self.max_epochs):
            dw, db  = 0.0, 0.0
            for x,y in zip(X,Y):
                dw += grad_w(w,b,x,y)
                db += grad_b(w,b,x,y)
                w = w - eta*dw
                b = b - eta*db
        return w, b       
        
        
        
    def do_mini_batch_gradient_descent(self):

        w, b , eta = init_w, init_b, 1.0
        mini_batch, number_of_points_seen = 2.0, 0
        for i in range(self.max_epochs):
            dw, db  = 0.0, 0.0
            for x,y in zip(X,Y):
                dw += grad_w(w,b,x,y)
                db += grad_b(w,b,x,y)
                number_of_points_seen += 1
                
                if number_of_points_seen % mini_batch == 0:
                    w = w - eta*dw
                    b = b - eta*db
                    dw, db = 0.0, 0.0 # resetting gradients - why though?
        return w, b   
            
            
