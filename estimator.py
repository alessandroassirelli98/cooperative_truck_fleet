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
        
        A = cas.jacobian(vehicle.f, S)
        G = cas.jacobian(vehicle.f, Nu)
        self.A_fun = cas.Function('A_fun', [S, U], [A])
        self.G_fun = cas.Function('G_fun', [S, U], [G])
        self.update_sensor_set() # Compute H matrix according to the available sensors

        self.Q = vehicle.Q
        self.R = vehicle.R
        
        S0_hat = vehicle.S0 * 0
        self.S_hat = np.zeros((self.n, N))
        self.P = np.zeros((self.n, self.n, N))

        # Initialization
        self.P[:,:,0] = np.eye(self.n) * 1e0
        self.S_hat[:,0] = S0_hat

        

    def update_sensor_set(self):
        H = cas.jacobian(self.vehicle.h, self.vehicle.S_sym)
        self.H_fun = cas.Function('H_fun', [self.vehicle.S_sym], [H])
        self.h_fun = cas.Function('h_fun', [self.vehicle.S_sym], [self.vehicle.h])

    
    def run_filter(self, u, i):
        self.update_sensor_set()
        G = self.G_fun(self.S_hat[:,i], u).full()
        A = self.A_fun(self.S_hat[:,i], u).full()
    
        # Prediction step
        self.S_hat[:,i+1] = self.vehicle.f_fun(self.S_hat[:,i], u, [0,0]).full().flatten()
        self.P[:,:, i+1] = A @ self.P[:,:,i] @ A.T + G @ self.Q @ G.T

        # Update step
        H = self.H_fun(self.S_hat[:,i+1]).full()
        # z = np.array([self.vehicle.S[1],
        #             self.vehicle.S[2],
        #             self.vehicle.S[3],
        #             self.vehicle.S[4],
        #             self.vehicle.S[5],
        #             self.vehicle.S[2] + self.vehicle.street.angle,
        #             self.vehicle.S[0]*np.cos(self.vehicle.street.angle) - self.vehicle.S[1]*np.sin(self.vehicle.street.angle) ,
        #             self.vehicle.S[0]*np.sin(self.vehicle.street.angle) + self.vehicle.S[1]*np.cos(self.vehicle.street.angle)
        #             ]) + self.eps[:,i] # Measurement Simulation
        
        z = self.h_fun(self.vehicle.S).full().flatten() + self.vehicle.eps[:,i] # Measurement Simulation

        S = H @ self.P[:,:,i+1] @ H.T + self.R
        w = self.P[:,:,i+1] @ H.T @ np.linalg.inv(S)
        self.S_hat[:,i+1] = self.S_hat[:,i+1] + (w @ (z.T - self.h_fun(self.S_hat[:,i+1]).full().flatten()))
        self.P[:,:,i+1] =  (np.eye(self.n) - w @ H) @ self.P[:,:,i+1]

