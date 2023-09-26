import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import casadi as cas
import conf
plt.style.use('seaborn')

class Estimator:
    def __init__(self, vehicle, dt, N):
        self.vehicle = vehicle

        self.dt = dt
        self.n = 6
        S = vehicle.S_sym
        U = vehicle.U_sym
        Nu = vehicle.Nu_sym
        
        A = cas.jacobian(vehicle.f, vehicle.S_sym)
        G = cas.jacobian(vehicle.f, vehicle.Nu_sym)
        self.A_fun = cas.Function('A_fun', [S, U, Nu], [A])
        self.G_fun = cas.Function('G_fun', [S, U, Nu], [G])

        self.Q = conf.Q
        self.R = conf.R
        self.H = conf.H
        S0 = vehicle.S0
        self.S_hat = np.zeros((self.n, N))
        self.P = np.zeros((self.n, self.n, N))

        # Initialization
        self.P[:,:,0] = np.eye(self.n) * 0
        self.S_hat[:,0] = S0

        self.nu = np.random.multivariate_normal([0, 0], self.Q, N).T
        self.eps = np.random.multivariate_normal(np.zeros(self.n), self.R, N).T

    
    def run_filter(self, u, i):
        G = self.G_fun(self.S_hat[:,i], u, self.nu[:,i]).full()
        A = self.A_fun(self.S_hat[:,i], u, self.nu[:,i]).full()
    
        # Prediction step
        self.S_hat[:,i+1] = self.vehicle.f_fun(self.S_hat[:,i], u, self.nu[:,i]).full().flatten()
        self.P[:,:, i+1] = A @ self.P[:,:,i] @ A.T + G @ self.Q @ G.T

        # Update step
        z = self.H @ self.vehicle.S + self.eps[:,i] # Measurement

        S = self.H @ self.P[:,:,i+1] @ self.H.T + self.R
        w = self.P[:,:,i+1] @ self.H.T @ np.linalg.inv(S)
        self.S_hat[:,i+1] = self.S_hat[:,i+1] + (w @ (z.T - self.H @ self.S_hat[:,i+1]))
        self.P[:,:,i+1] =  (np.eye(self.n) - w @ self.H) @ self.P[:,:,i+1]

