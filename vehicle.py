import numpy as np
import conf
from street import Street, Lane
from pid import PID
from ppc import PPC
from estimator import Estimator
import casadi as cas

class Vehicle:
    def __init__(self, street:Street, lane:Lane, s, v, dt, N, L=3):
        self.pid_steer = PID(kp=conf.kp_pos, kd=conf.kd_pos, ki=conf.ki_pos, integral_sat=conf.v_max)
        self.pid_vel = PID(kp=conf.kp_vel, kd=conf.kd_vel, ki=conf.ki_vel, integral_sat=conf.v_max)
        
        self.path = np.array([[lane.x_start, lane.y_start], [lane.x_end, lane.y_end]])
        self.steer_control = PPC()
        self.steer_control.lookAheadDistance = 20


        self.dt = dt
        self.street = street
        self.lane = lane
        self.s = s
        self.s_hat = np.array([s])
        self.x, self.y = lane.s_to_xy(s)
        self.delta = 0
        self.alpha = 0
        self.v = v
        self.omega = 0
        self.a = 0
        self.u = 0

        self.S0 = np.array([self.x, self.y, self.delta, self.alpha, v, self.a])
        self.S = self.S0

        self.L = L # Wheelbase of vehicle
        self.tau = conf.tau # Time constant of the acceleration
        self.r = conf.r # parameters that indicates standstill spacing
        self.h = conf.h # parameters that indicates spacing between front vehicles d  = h * v

        self.v_max = conf.v_max
        self.a_max = conf.a_max

        # self.q = np.array([self.x, self.y, self.delta, self.alpha])
        # self.dq = np.array([np.cos(self.delta) * v,
        #                     np.sin(self.delta) * v, 
        #                     np.tan(self.alpha)/L * v, 
        #                     0])
        
        # Model
        # Thir order model of a car
        # x, y, delta, alpha, v1, a1
        # Where delta is the rotation of the vehicle, alpha the steering angle
        # v1 is the forward velocity,
        # a1 is the forward acceleration
        # it is third order because the acceleration is linked to the engine input u1
        self.S_sym = cas.SX.sym('S', 6)
        self.Nu_sym = cas.SX.sym('Nu', 2)
        self.U_sym = cas.SX.sym('U', 2)

        delta = self.S_sym[2]
        alpha = self.S_sym[3]
        v1 = self.S_sym[4]
        a1 = self.S_sym[5]

        # The nonlinear model (Obtained by Euler discretization)
        self.f = cas.vertcat(cas.cos(delta) * v1,
                        cas.sin(delta) * v1,
                        cas.tan(alpha)/L * v1,
                        self.U_sym[1] + self.Nu_sym[1],
                        a1,
                        1/self.tau * (-a1 + (self.U_sym[0] + self.Nu_sym[0]))
                    ) * dt + self.S_sym
        
        self.f_fun = cas.Function('f_fun', [self.S_sym, self.U_sym, self.Nu_sym], [self.f])

        self.estimator = Estimator(self, dt, N)
        
        
        self.e1 = 0 
        self.beta = 0 # Angle of the street at the current position
        self.M0Street = np.matrix([[1, 0], [0, 1]]) # Rotation matrix of the street 

        # 100 l
        # 1 l/10 km
        # v = 10 km/h
        # t = 1h
        # Autonomy is given in liters of fuel
        # C0 is the fuel consumption expressed in l per km
        # C1 is the additional fuel consumption in l per km if the vehicle is the leader
        self.autonomy = 100
        self.c0 = 1 
        self.c1 = 0.5

        self.leader = False
        self.init = False
        self.cnt = 0

        self.log_u = []
        self.log_s = []
        self.log_s_hat = []
        self.log_v = []
        self.log_e = []
        self.log_path = []
        self.log_xydelta = []


    def track_front_vehicle(self, front_vehicle, use_velocity_info = True):
        self.e1 = front_vehicle.s - self.s - self.r - self.h * self.v
        self.e2 = front_vehicle.v - self.v - self.h * self.a
        self.e3 = front_vehicle.a - self.a - (1/conf.tau * (- self.a + self.u) * self.h)

        if use_velocity_info:
            self.u += 1/self.h * ( - self.u + conf.kp*self.e1 + conf.kd*self.e2 + conf.kdd*self.e3 + front_vehicle.u) * self.dt
        else:
            self.u += 1/self.h * ( - self.u + conf.kp*self.e1 + conf.kd*self.e2 + conf.kdd*self.e3) * self.dt
        self.omega = 0
    

    def set_desired_velocities(self, v_des):
        self.u = self.pid_vel.compute(v_des - self.v, self.dt)

    def measure_street(self):
        self.beta = self.street.angle + np.random.randn() * conf.sigma_street_angle **2
        self.M0Street = np.matrix([[np.cos(self.beta), -np.sin(self.beta)], [np.sin(self.beta), np.cos(self.beta)]])


    def change_lane(self, lane: Lane):
        self.lane = lane
        target = np.matrix([[self.x], [self.y]]) + (self.M0Street @ np.matrix([[self.street.lane_width], [self.street.lane_width]]))
        x_target = target[0,0]
        y_target = target[1,0]
        self.path = np.array([[self.x, self.y], [x_target, y_target], [lane.x_end, lane.y_end]])


    def update(self):
        self.measure_street()

        xy_position = np.array([self.x, self.y])
        alpha_des = self.steer_control.ComputeSteeringAngle(self.path, xy_position, self.delta, self.L)
        self.omega = self.pid_steer.compute(alpha_des - self.alpha, self.dt)  # Compute steering angle acceleration
        

        u = np.array([self.u, self.omega])
        self.S = self.f_fun(self.estimator.S_hat[:, self.cnt], u, [0,0])
        self.S = self.S.full().flatten() # Convert to numpy array
        self.S[5] = np.clip(self.S[5], -self.a_max, self.a_max) # Clip acceleration

        self.estimator.run_filter(u, self.cnt)

        self.x = self.estimator.S_hat[0, self.cnt]
        self.y = self.estimator.S_hat[1, self.cnt]
        self.delta = self.estimator.S_hat[2, self.cnt]
        self.alpha = self.estimator.S_hat[3, self.cnt]
        self.v = self.estimator.S_hat[4, self.cnt]
        self.a = self.estimator.S_hat[5, self.cnt]

        self.s = self.street.xy_to_s(self.x, self.y)
        
        self.log_u.append(self.u)
        self.log_s.append(self.s)
        self.log_s_hat.append(self.s_hat[0])
        self.log_v.append(self.v)
        self.log_e.append(alpha_des - self.alpha)
        self.log_path.append(self.path)
        self.log_xydelta.append([self.S[0], self.S[1], self.S[2]])
        self.cnt += 1