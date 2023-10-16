import numpy as np
import matplotlib.pyplot as plt
import casadi as cas
import conf
plt.style.use('seaborn')

class Estimator:
    def __init__(self, vehicle, dt, N):
        self.vehicle = vehicle
        self.vehicles_list = []
        self.n_vehicles = 0

        self.dt = dt
        self.n = 7
        self.N = N
        
        self.create_sys_model()
        self.create_measurement_model()
        
        
        S0_hat = vehicle.S0#np.zeros(self.n)
        self.S_hat = np.zeros((self.n, N))
        self.S = np.zeros((self.n, N))
        self.S[:,0] = vehicle.S0
        self.P = np.zeros((self.n, self.n, N))

        # Initialization
        self.P[:,:,0] = np.eye(self.n)*1e-5
        self.S_hat[:,0] = S0_hat


    def create_measurement_model(self):
        x = self.S_sym[0]
        y = self.S_sym[1]
        delta = self.S_sym[2]
        alpha = self.S_sym[3]
        v1 = self.S_sym[4]
        beta = self.S_sym[6]
        


        self.h = cas.vertcat(x, # GPS
                            y, # GPS
                            delta, # Magnetometer
                            alpha, # Encoder on steering wheel
                            v1, # Encoder on motor shaft
                            (- cas.sin(beta)*x + cas.cos(beta)*y), # Lane detection with cameras
                        )
        
        self.R = np.diag([
            conf.sigma_x_gps**2, 
            conf.sigma_y_gps**2,
            conf.sigma_delta**2, 
            conf.sigma_alpha**2, 
            conf.sigma_v**2, 
            conf.sigma_y**2
            ])
        

        if len(self.vehicles_list) != 0:
            v = self.vehicles_list[0]  
            x_t = self.S_sym[7]
            y_t = self.S_sym[8]
            delta = self.S_sym[2]
            xt_1 = x_t * cas.cos(delta) + y_t * cas.sin(delta) - x * cas.cos(delta) - y * cas.sin(delta)
            yt_1 = - x_t * cas.sin(delta) + y_t * cas.cos(delta) + x * cas.sin(delta) - y * cas.cos(delta)

            self.h = cas.vertcat(self.h,
                                cas.vertcat(
                                    cas.sqrt(xt_1**2 + yt_1**2), # Lidar
                                    cas.arctan(yt_1 / xt_1), # Lidar
                                    0*x_t,
                                    0*y_t
                                ))
            self.R = np.diag([
            conf.sigma_x_gps**2, 
            conf.sigma_y_gps**2,
            conf.sigma_delta**2, 
            conf.sigma_alpha**2, 
            conf.sigma_v**2, 
            conf.sigma_y**2,
            conf.sigma_lidar_rho**2,
            conf.sigma_lidar_phi**2,
            conf.sigma_x_gps**2, 
            conf.sigma_y_gps**2
            ])


        H = cas.jacobian(self.h, self.S_sym)
        self.H_fun = cas.Function('H_fun', [self.S_sym], [H])
        self.h_fun = cas.Function('h_fun', [self.S_sym], [self.h])

    def measure(self, i):

        eps = np.random.multivariate_normal(np.zeros(self.R.shape[0]), self.R)

        if len(self.vehicles_list) != 0:
            v = self.vehicles_list[0]   
            x = v.S[0]
            y = v.S[1]
            Xf_0 = np.array([x,y,1])
            Xf_1 = self.vehicle.M10 @ Xf_0
            xf_1 = Xf_1[0]
            yf_1 = Xf_1[1]

            rho = np.sqrt(xf_1**2 + yf_1**2)
            phi = np.arctan(yf_1 / xf_1)

            Xf_0 = self.vehicle.M01 @ np.array([rho * np.cos(phi), rho*np.sin(phi), 1])

            z = np.array([self.vehicle.S[0],
                self.vehicle.S[1],
                self.vehicle.S[2],
                self.vehicle.S[3],
                self.vehicle.S[4],
                -self.vehicle.S[0]*np.sin(self.vehicle.street.angle) + self.vehicle.S[1]*np.cos(self.vehicle.street.angle),
                rho,
                phi,
                x,
                y]) + eps # Measurement Simulation
        else:
            z = np.array([self.vehicle.S[0],
                    self.vehicle.S[1],
                    self.vehicle.S[2],
                    self.vehicle.S[3],
                    self.vehicle.S[4],
                    -self.vehicle.S[0]*np.sin(self.vehicle.street.angle) + self.vehicle.S[1]*np.cos(self.vehicle.street.angle),
                    ]) + eps # Measurement Simulation
            
        return z
    
    def update_vehcles_list(self):
        self.vehicles_list = self.vehicle.trucks_visible
        self.vehicles_list = self.vehicles_list
        self.n_vehicles = len(self.vehicles_list)
        self.create_sys_model()
        self.create_measurement_model()

    
    def run_filter(self, u, i):
        self.S[:,i + 1] = self.vehicle.S # Save the real state
        z = self.measure(i)
        if self.n_vehicles != 0:
            u = cas.vertcat(u, self.vehicles_list[0].v, self.vehicles_list[0].omega)

        nu = np.random.multivariate_normal(np.zeros(self.Q.shape[0]), self.Q)

        G = self.G_fun(self.S_hat[:,i], u).full()
        A = self.A_fun(self.S_hat[:,i], u, np.zeros(2 + 2 *self.n_vehicles)).full()
    
        # Prediction step
        self.S_hat[:,i+1] = self.f_fun(self.S_hat[:,i], u, nu).full().flatten()
        self.P[:,:, i+1] = A @ self.P[:,:,i] @ A.T + G @ self.Q @ G.T

        # Update step
        H = self.H_fun(self.S_hat[:,i+1]).full()

        S = H @ self.P[:,:,i+1] @ H.T + self.R
        w = self.P[:,:,i+1] @ H.T @ np.linalg.inv(S)
        self.S_hat[:,i+1] = self.S_hat[:,i+1] + (w @ (z.T - self.h_fun(self.S_hat[:,i+1]).full().flatten()))
        self.P[:,:,i+1] =  (np.eye(self.P.shape[0]) - w @ H) @ self.P[:,:,i+1]


    def create_sys_model(self):

        S = cas.SX.sym('S_aug', 4 * self.n_vehicles)
        U = cas.SX.sym('U_aug', 2 * self.n_vehicles)
        Nu = cas.SX.sym('Nu_aug', 2 * self.n_vehicles)

        self.S_sym = cas.vertcat(self.vehicle.S_sym, S)
        self.U_sym = cas.vertcat(self.vehicle.U_sym, U)
        self.Nu_sym = cas.vertcat(self.vehicle.Nu_sym, Nu)
    

        if len(self.vehicles_list) != 0:
            delta = self.S_sym[7 + 2]
            alpha = self.S_sym[7 + 3]
            v = self.U_sym[2]  + self.Nu_sym[2]
            omega = self.U_sym[3] + self.Nu_sym[3]

            self.f = cas.vertcat(self.vehicle.f, 
                                (cas.vertcat(cas.cos(delta) * v,
                                    cas.sin(delta) * v,
                                    cas.tan(alpha)/self.vehicle.L * v,
                                    omega) * self.dt + self.S_sym[7:]
                                )
                                )
            self.Q = np.diag([conf.sigma_u**2, conf.sigma_omega**2, 10 * conf.sigma_u**2, 10 * conf.sigma_omega**2])
            s_ = np.zeros((4, self.N))
            self.S_hat = np.concatenate([self.S_hat, s_])

            P_tmp = np.zeros((11,11,self.N))
            for i in range(self.N):
                P_tmp[:,:,i] = np.block([ [self.P[:,:, i], np.zeros((7,4))], [np.zeros((4, 7)), np.eye(4) * 1e-5] ])
            self.P = P_tmp
            
            
        else:
            self.f = self.vehicle.f
            self.Q = np.diag([conf.sigma_u**2, conf.sigma_omega**2])
        
        
        A = cas.jacobian(self.f, self.S_sym)
        G = cas.jacobian(self.f, self.U_sym)
        self.A_fun = cas.Function('A_fun', [self.S_sym, self.U_sym, self.Nu_sym], [A])
        self.G_fun = cas.Function('G_fun', [self.S_sym, self.U_sym], [G])
        self.f_fun = cas.Function('f_fun', [self.S_sym, self.U_sym, self.Nu_sym], [self.f])

