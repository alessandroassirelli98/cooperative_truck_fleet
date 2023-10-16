import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import casadi as cas
import conf
plt.style.use('seaborn')

class SurroundingEstimator:
    def __init__(self, vehicle, dt, N, vehicles_list = []):
        self.vehicle = vehicle
        self.dt = dt

        self.n_vehicles = len(vehicles_list)
        self.vehicles_list = vehicles_list

        self.R = np.diag([
            conf.sigma_radar_rho**2,
            conf.sigma_radar_phi**2,
            ])
        
        self.Q = np.diag([conf.sigma_u**2, conf.sigma_omega**2])
        self.nu = np.random.multivariate_normal([0, 0], self.Q, N).T
        self.eps = np.random.multivariate_normal(np.zeros(self.R.shape[0]), self.R, N).T

        S0_hat = np.zeros(4)
        self.S_hat = np.zeros((4, N))
        self.S = np.zeros((4, N))
        self.P = np.zeros((4,  4, N))

        # Initialization
        self.P[:,:,0] = np.eye(self.n)*1e-5
        self.S_hat[:,0] = S0_hat

        
    def update_vehcles_list(self, vehicles_list):
        self.vehicles_list = vehicles_list
        self.n_vehicles = len(vehicles_list)
        self.create_sys_model()
        self.update_measurement_model()


    def update_measurement_model(self):

        S = cas.SX.sym('S', 4 * self.n_vehicles)
        # Tracked vehicle coordinates in world rf
        x = S[0]
        y = S[1]

        x_1 = x * cas.cos(self.vehicle.delta) + y * cas.sin(self.vehicle.delta)\
            - self.vehicle.x * cas.cos(self.vehicle.delta) - self.vehicle.y * cas.sin(self.vehicle.delta)
        
        y_1 = - x * cas.sin(self.vehicle.delta) + y * cas.cos(self.vehicle.delta)\
                + self.vehicle.x * cas.sin(self.vehicle.delta) - self.vehicle.y * cas.cos(self.vehicle.delta)

        self.h = cas.vertcat(cas.sqrt(x_1**2 + y_1**2), cas.atan2(y_1, x_1))

        H = cas.jacobian(self.h, S)
        self.H_fun = cas.Function('H_fun', [S], [H])
        self.h_fun = cas.Function('h_fun', [S], [self.h])


    def measure(self, i):
        v = self.vehicles_list[0]
        x = v.S[0]
        y = v.S[1]
        Xf_0 = np.array([x,y,1])
        Xf_1 = self.vehicle.M10 @ Xf_0
        xf_1 = Xf_1[0]
        yf_1 = Xf_1[1]

        rho = np.sqrt(xf_1**2 + yf_1**2)
        phi = np.arctan2(yf_1, xf_1)

        z = np.array([rho, phi]) + self.eps[:,i]

        return z

    
    def run_filter(self, i):
        z = self.measure(i)
        
        G = self.G_fun(self.S_hat[:,i], u).full()
        A = self.A_fun(self.S_hat[:,i], u).full()
    
        # Prediction step
        self.S_hat[:,i+1] = self.f_fun(self.S_hat[:,i], u, [self.vehicles_list[0].v,self.vehicles_list[0].omega]).full().flatten()
        self.P[:,:, i+1] = A @ self.P[:,:,i] @ A.T + G @ self.Q @ G.T

        # Update step
        H = self.H_fun(self.S_hat[:,i+1]).full()

        S = H @ self.P[:,:,i+1] @ H.T + self.R
        w = self.P[:,:,i+1] @ H.T @ np.linalg.inv(S)
        self.S_hat[:,i+1] = self.S_hat[:,i+1] + (w @ (z.T - self.h_fun(self.S_hat[:,i+1]).full().flatten()))
        self.P[:,:,i+1] =  (np.eye(4) - w @ H) @ self.P[:,:,i+1]


    def create_sys_model(self):
        # Create the system model of the other vehicles
        self.S_sym = cas.SX.sym('S', 4 * self.n_vehicles)
        self.U_sym = cas.SX.sym('U', 2 * self.n_vehicles)
        self.Nu_sym = cas.SX.sym('Nu', 2 * self.n_vehicles)


        delta = self.S_sym[2]
        alpha = self.S_sym[3]
        v = self.U_sym[0]  + self.Nu_sym[0]
        omega = self.U_sym[1] + self.Nu_sym[1]
        self.f = cas.vertcat(cas.cos(delta) * v,
                    cas.sin(delta) * v,
                    cas.tan(alpha)/self.L * v,
                    omega) * self.dt + self.S_sym
        
        A = cas.jacobian(self.vehicle.f, self.S_sym)
        G = cas.jacobian(self.vehicle.f, self.Nu_sym)
        self.A_fun = cas.Function('A_fun', [self.S_sym, self.U_sym], [A])
        self.G_fun = cas.Function('G_fun', [self.S_sym, self.U_sym], [G])