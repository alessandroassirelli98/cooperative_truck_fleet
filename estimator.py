import numpy as np
import matplotlib.pyplot as plt
import casadi as cas
import conf
plt.style.use('seaborn')

class Estimator:
    def __init__(self, vehicle, dt, N):
        
        self.vehicle = vehicle

        self.dt = dt
        self.n = 7
        self.N = N

        self.visible_vehicles = []
        self.visible_vehicles_storage = []
        self.not_removed_vehicles = []
        self.new_vehicles = []
        self.n_other_vehicles = 6
        self.n_vehicles = 0
        self.n_vehicles_storage = 0
        
        self.S0_hat = vehicle.S0 #np.zeros(self.n)
        self.S_hat = np.zeros((self.n, N))
        self.S = np.zeros((self.n, N))
        self.S[:,0] = vehicle.S0
        self.P = np.zeros((self.n, self.n, N))

        self.create_sys_model()
        self.create_measurement_model()

        # Initialization
        self.P[:,:,0] = np.eye(self.n)*1e-5
        self.S_hat[:,0] = self.S0_hat


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
        
        h_tmp = []
        R_tmp = [conf.sigma_x_gps**2, 
                conf.sigma_y_gps**2,
                conf.sigma_delta**2, 
                conf.sigma_alpha**2, 
                conf.sigma_v**2, 
                conf.sigma_y**2]
        
        for i, v in enumerate(self.visible_vehicles_storage):
            visibility = 1 if v in self.visible_vehicles else 0

            x_o_0 = self.S_sym[self.n + self.n_other_vehicles * i + 0]
            y_o_0 = self.S_sym[self.n + self.n_other_vehicles * i + 1]
            x_o_1 = x_o_0 * cas.cos(delta) + y_o_0 * cas.sin(delta) - x * cas.cos(delta) - y * cas.sin(delta)
            y_o_1 = - x_o_0 * cas.sin(delta) + y_o_0 * cas.cos(delta) + x * cas.sin(delta) - y * cas.cos(delta)

            h_tmp.append(visibility * cas.vertcat(cas.sqrt(x_o_1**2 + y_o_1**2),
                                                cas.arctan(y_o_1 / x_o_1)
                                                ))
            
            R_tmp.append(conf.sigma_lidar_rho**2 * visibility)
            R_tmp.append(conf.sigma_lidar_phi**2 * visibility)
            # R_tmp.append(conf.sigma_x_gps**2 * 0)
            # R_tmp.append(conf.sigma_y_gps**2 * 0)

        self.h = self.h
        for e in h_tmp: self.h = cas.vertcat(self.h, e)
        self.R = np.diag(R_tmp)

        H = cas.jacobian(self.h, self.S_sym)
        self.H_fun = cas.Function('H_fun', [self.S_sym], [H])
        self.h_fun = cas.Function('h_fun', [self.S_sym], [self.h])

    def measure(self, i):

        eps = np.random.multivariate_normal(np.zeros(self.R.shape[0]), self.R)
        z_tmp = [self.vehicle.S[0],
                self.vehicle.S[1],
                self.vehicle.S[2],
                self.vehicle.S[3],
                self.vehicle.S[4],
                -self.vehicle.S[0]*np.sin(self.vehicle.street.angle) + self.vehicle.S[1]*np.cos(self.vehicle.street.angle)]
        
        for i, v in enumerate(self.visible_vehicles_storage):
            visibility = 1 if v in self.visible_vehicles else 0
            x = v.S[0]
            y = v.S[1]
            Xf_0 = np.array([x,y,1])
            Xf_1 = self.vehicle.M10 @ Xf_0
            xf_1 = Xf_1[0]
            yf_1 = Xf_1[1]

            rho = np.sqrt(xf_1**2 + yf_1**2)
            phi = np.arctan(yf_1 / xf_1)

            Xf_0 = self.vehicle.M01 @ np.array([rho * np.cos(phi), rho*np.sin(phi), 1])

            z_tmp.append(visibility*rho)
            z_tmp.append(visibility*phi)
            # z_tmp.append(0*x)
            # z_tmp.append(0*y)

        z = np.array(z_tmp) + eps # Measurement Simulation
            
        return z
    
    def update_vehcles_list(self):
        self.visible_vehicles = self.vehicle.trucks_visible.copy()

        self.new_vehicles = []
        for v in self.visible_vehicles:
            if v not in self.visible_vehicles_storage:
                self.new_vehicles.append(v)
        
        for _, v in enumerate(self.visible_vehicles):
            if v not in self.visible_vehicles_storage: 
                self.visible_vehicles_storage.append(v)

        self.n_vehicles = len(self.visible_vehicles)
        self.n_vehicles_storage = len(self.visible_vehicles_storage)

        self.create_sys_model()
        self.create_measurement_model()

    
    def run_filter(self, u, i):
        self.S[:,i + 1] = self.vehicle.S.copy() # Save the real state
        z = self.measure(i)
        u = [u[0], u[1]]
        for v in self.visible_vehicles_storage:
            u.append(v.u[0])
            u.append(v.u[1])
        u = np.array(u)

        nu = np.random.multivariate_normal(np.zeros(self.Q.shape[0]), self.Q)

        G = self.G_fun(self.S_hat[:,i], u).full()
        A = self.A_fun(self.S_hat[:,i], u, np.zeros(2 + 2 *self.n_vehicles_storage)).full()
    
        # Prediction step
        self.S_hat[:,i+1] = self.f_fun(self.S_hat[:,i], u, nu).full().flatten()
        self.P[:,:, i+1] = A @ self.P[:,:,i] @ A.T + G @ self.Q @ G.T

        # Update step
        H = self.H_fun(self.S_hat[:,i+1]).full()

        S = H @ self.P[:,:,i+1] @ H.T + self.R
        w = self.P[:,:,i+1] @ H.T @ np.linalg.pinv(S)
        self.S_hat[:,i+1] = self.S_hat[:,i+1] + (w @ (z.T - self.h_fun(self.S_hat[:,i+1]).full().flatten()))
        self.P[:,:,i+1] =  (np.eye(self.P.shape[0]) - w @ H) @ self.P[:,:,i+1]

    def create_sys_model(self):

        S = cas.SX.sym('S_aug', self.n_other_vehicles * self.n_vehicles_storage)
        U = cas.SX.sym('U_aug', 2 * self.n_vehicles_storage)
        Nu = cas.SX.sym('Nu_aug', 2 * self.n_vehicles_storage)

        self.S_sym = cas.vertcat(self.vehicle.S_sym, S)
        self.U_sym = cas.vertcat(self.vehicle.U_sym, U)
        self.Nu_sym = cas.vertcat(self.vehicle.Nu_sym, Nu)
    
        tmp_f = []
        Q_tmp = [conf.sigma_u**2, conf.sigma_omega**2]
        for i, v in enumerate(self.visible_vehicles_storage):
            visibility = 1 #if v in self.visible_vehicles else 0
            delta_other = self.S_sym[self.n + self.n_other_vehicles * i + 2]
            alpha_other = self.S_sym[self.n + self.n_other_vehicles * i + 3]
            v_other = self.S_sym[self.n + self.n_other_vehicles * i + 4]
            a_other = self.S_sym[self.n + self.n_other_vehicles * i + 5]
            u_other = self.U_sym[2 * i + 2]  + self.Nu_sym[2 * i + 2]
            omega_other = self.U_sym[2 * i + 3] + self.Nu_sym[2 * i + 3]
            tmp_f.append(visibility*(cas.vertcat(cas.cos(delta_other) * v_other,
                                    cas.sin(delta_other) * v_other,
                                    cas.tan(alpha_other)/self.vehicle.L * v_other,
                                    omega_other,
                                    a_other,
                                    1/self.vehicle.tau * (-a_other + (u_other)))
                                    * self.dt + self.S_sym[7 + i*self.n_other_vehicles: 7 + (i+1)*self.n_other_vehicles]
                                    )
                        )
            
            Q_tmp.append(10 * conf.sigma_u**2)
            Q_tmp.append(10 * conf.sigma_omega**2)

        self.f = self.vehicle.f
        for e in tmp_f: self.f = cas.vertcat(self.f, e)

        self.Q = np.diag(Q_tmp)

        # slices_to_keep = np.arange(0, 7)
        # for i, _ in self.not_removed_vehicles:
        #     slices_to_keep = np.append(slices_to_keep, np.arange(self.n + self.n_other_vehicles * i, self.n + self.n_other_vehicles * (i+1)))

        s_tmp = np.zeros((self.n_other_vehicles * len(self.new_vehicles), self.N))
        self.S_hat = np.concatenate([self.S_hat, s_tmp])

        P_tmp = np.zeros((self.n + self.n_vehicles_storage * self.n_other_vehicles , self.n + self.n_vehicles_storage * self.n_other_vehicles, self.N))
        P_other = np.eye(len(self.new_vehicles) * self.n_other_vehicles) * 1e-5
        for i in range(self.N):
            P_tmp[:,:,i] = np.block([ [self.P[:,:, i], np.zeros((self.P.shape[0], len(self.new_vehicles) * self.n_other_vehicles))],
                                    [np.zeros((self.n_other_vehicles * len(self.new_vehicles), self.P.shape[0])), P_other ] ])
            
        self.P = P_tmp
        
        
        A = cas.jacobian(self.f, self.S_sym)
        G = cas.jacobian(self.f, self.U_sym)
        self.A_fun = cas.Function('A_fun', [self.S_sym, self.U_sym, self.Nu_sym], [A])
        self.G_fun = cas.Function('G_fun', [self.S_sym, self.U_sym], [G])
        self.f_fun = cas.Function('f_fun', [self.S_sym, self.U_sym, self.Nu_sym], [self.f])

