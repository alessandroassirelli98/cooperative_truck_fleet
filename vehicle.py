from collections import OrderedDict
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
        
        self.path = np.array([[0, 0], [lane.length, 0]])
        self.steer_control = PPC()
        self.steer_control.lookAheadDistance = 20

        self.dt = dt
        self.street = street
        self.lane = lane
        self.x, self.y = s, 0
        self.delta = 0
        self.alpha = 0
        self.v = v
        self.v_des = v
        self.omega = 0
        self.a = 0
        self.u_fwd = 0
        self.u = np.array([self.u_fwd, self.omega])
        

        self.S0 = np.array([self.x, self.y, self.delta, self.alpha, v, self.a])
        self.S = self.S0

        self.L = L # Wheelbase of vehicle
        self.tau = conf.tau # Time constant of the acceleration
        self.r = conf.r # parameters that indicates standstill spacing

        self.v_max = conf.v_max
        self.a_max = conf.a_max

        
        # Model
        # The model is expressed in the street reference frame, where the x axis is the street axis
        # The street reference frame in the simple case is centered in the fixed one, so to retrieve
        # the position of the vehicle in the world reference frame, you just need to R01.@(x, y)

        # Third order model of a car
        # x, y, delta, alpha, v1, a1, beta
        # Where delta is the rotation of the vehicle, alpha the steering angle
        # v1 is the forward velocity,
        # a1 is the forward acceleration
        # beta is the angle of the street at the current position
        # it is third order because the acceleration is linked to the engine input u1
        self.S_sym = cas.SX.sym('S', 6)
        self.Nu_sym = cas.SX.sym('Nu', 2)
        self.U_sym = cas.SX.sym('U', 2)

        x = self.S_sym[0]
        y = self.S_sym[1]
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
                        1/self.tau * (-a1 + (self.U_sym[0] + self.Nu_sym[0])),
                    ) * dt + self.S_sym
        
        self.f_fun = cas.Function('f_fun', [self.S_sym, self.U_sym, self.Nu_sym], [self.f])

        # The measurement model
        self.h = cas.vertcat(y, # Lane detection with cameras
                             delta, # Cameras
                             delta + street.angle, # Magnetometer
                             alpha, # Encoder on steering wheel
                             v1, # Encoder on motor shaft
                             a1, # Accelerometer
                             (x * cas.cos(street.angle) - y * cas.sin(street.angle)), # GPS
                             (x * cas.sin(street.angle) + y * cas.cos(street.angle)), # GPS
                             0
                            )
        
        self.R = np.diag([conf.sigma_y**2,
            conf.sigma_delta**2, 
            conf.sigma_mag**2 + conf.sigma_beta**2, 
            conf.sigma_alpha**2, 
            conf.sigma_v**2, 
            conf.sigma_a**2, 
            conf.sigma_x_gps**2 + conf.sigma_y_gps**2, 
            conf.sigma_x_gps**2 + conf.sigma_y_gps**2,
            conf.sigma_radar**2
            ])
        self.Q = conf.Q       
        self.nu = np.random.multivariate_normal([0, 0], self.Q, N).T
        self.eps = np.random.multivariate_normal(np.zeros(self.R.shape[0]), self.R, N).T
        self.u_unc = self.u + self.nu[:, 0]

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
        self.status = [self.autonomy, self.c0, self.c1, self.x]
        self.platoon_status = OrderedDict()
        self.platoon_status["leader"] = None
        self.platoon_status["overtaking"] = None
        self.schedule = OrderedDict()

        self.in_overtake = False
        self.plan_overtake = False
        self.leader = False
        self.init = False
        self.cnt = 0

        self.log_u = []
        self.log_u_unc = []
        self.log_x = []
        self.log_x_true = []
        self.log_v = []
        self.log_e = []
        self.log_path = []
        self.log_xydelta = []
        self.log_xydelta_world = []


    def track_front_vehicle(self, front_vehicle, use_velocity_info = True):
        self.e1 = front_vehicle.x - self.x - self.r - conf.h * self.v
        self.e2 = front_vehicle.v - self.v - conf.h * self.a
        self.e3 = front_vehicle.a - self.a - (1/conf.tau * (- self.a + self.u_fwd) * conf.h)

        if use_velocity_info:
            self.u_fwd += 1/conf.h * ( - self.u_fwd + 
                                  conf.kp*self.e1 + 
                                  conf.kd*self.e2 + 
                                  conf.kdd*self.e3 + 
                                  front_vehicle.u_fwd) * self.dt
        else:
            self.u_fwd += 1/conf.h * ( - self.u + conf.kp*self.e1 + conf.kd*self.e2 + conf.kdd*self.e3) * self.dt
        self.omega = 0  



    def set_desired_velocities(self, v_des):
        self.u_fwd = self.pid_vel.compute(v_des - self.v, self.dt)

    def overtake(self):
            leader = self.platoon_status["leader"]
            v_leader = leader.v
            x_leader = leader.x

            if self.x < x_leader + 10:
                self.lane = self.street.lanes[1]
                x_target = self.x + self.street.lane_width
                y_target = self.street.lane_width
                self.v_des = v_leader + 10
                self.path = np.array([[self.x, self.y], 
                                  [x_target, y_target], 
                                  [self.lane.x_end, self.street.lane_width]]) # Change lane
                self.in_overtake = True
                self.platoon_status["overtaking"] = self

            else: # When the vehicle is in front of the leader go back to lane 0
                self.lane = self.street.lanes[0]
                x_target = self.x + self.street.lane_width
                y_target = 0
                self.path = np.array([[self.x, self.y], 
                                  [x_target, y_target], 
                                  [self.lane.x_end, self.street.lane_width]]) # Change lane

                self.v_des -= 10
                self.in_overtake = False
                self.platoon_status["overtaking"] = None

            self.plan_overtake = False

                
    def update(self, front_vehicle = None):
        self.update_sensors(front_vehicle)
        try:
            if self.x - self.x_received_schedule > self.schedule[self][0] \
                and self.platoon_status["overtaking"] is None and not self.leader: 
                self.plan_overtake = True
                del self.schedule[self]
            else:
                self.plan_overtake = False
        except:
            self.plan_overtake = False
        
        if self.plan_overtake or self.in_overtake:
            self.overtake()

        if not self.leader and not self.in_overtake:
            self.track_front_vehicle(front_vehicle)
        else:
            self.u_fwd = self.pid_vel.compute(self.v_des - self.v, self.dt)

        xy_position = np.array([self.x, self.y])
        alpha_des = self.steer_control.ComputeSteeringAngle(self.path, xy_position, self.delta, self.L)
        self.omega = self.pid_steer.compute(alpha_des - self.alpha, self.dt)  # Compute steering angle acceleration

        

        self.u = np.array([self.u_fwd, 0])
        self.u_unc = self.u + self.nu[:, self.cnt].T

        self.estimator.run_filter(self.u_unc, self.cnt)

        self.S = self.f_fun(self.S, self.u, [0,0])
        self.S = self.S.full().flatten() # Convert to numpy array
        self.S[5] = np.clip(self.S[5], -self.a_max, self.a_max) # Clip acceleration


        self.x = self.estimator.S_hat[0, self.cnt + 1]
        self.y = self.estimator.S_hat[1, self.cnt + 1]
        self.delta = self.estimator.S_hat[2, self.cnt + 1]
        self.alpha = self.estimator.S_hat[3, self.cnt + 1]
        self.v = self.estimator.S_hat[4, self.cnt + 1]
        self.a = self.estimator.S_hat[5, self.cnt + 1]

        consumption = self.c0 + self.c1 if self.leader else self.c0
        self.autonomy -= consumption * self.v * self.dt
        self.status = [self.autonomy , self.c0, self.c1, self.x]
        
        self.log_u.append(self.u)
        self.log_u_unc.append(self.u_unc)
        self.log_x.append(self.x)
        self.log_x_true.append(self.S[0])
        self.log_v.append(self.v)
        self.log_e.append(self.e1)
        self.log_path.append(self.path)
        self.log_xydelta.append([self.S[0], self.S[1], self.S[2]])
        self.log_xydelta_world.append([self.S[0] * np.cos(self.street.angle) - self.S[1] * np.sin(self.street.angle),
                                       self.S[0] * np.sin(self.street.angle) + self.S[1] * np.cos(self.street.angle),
                                    self.S[2] + self.street.angle])
        self.cnt += 1


    def update_sensors(self, front_vehicle):
        x = self.S_sym[0]
        y = self.S_sym[1]
        delta = self.S_sym[2]
        alpha = self.S_sym[3]
        v1 = self.S_sym[4]
        a1 = self.S_sym[5]
        if not self.leader:
            self.h = cas.vertcat(y, # Lane detection with cameras
                             delta, # Cameras
                             delta + self.street.angle, # Magnetometer
                             alpha, # Encoder on steering wheel
                             v1, # Encoder on motor shaft
                             a1, # Accelerometer
                             0*(x * cas.cos(self.street.angle) - y * cas.sin(self.street.angle)), # GPS
                             0*(x * cas.sin(self.street.angle) + y * cas.cos(self.street.angle)), # GPS
                             x # Radar
                            )
            self.R[-1,-1] = conf.sigma_radar**2 + front_vehicle.estimator.P[0, 0, self.cnt]

        else:
            self.h = cas.vertcat(y, # Lane detection with cameras
                             delta, # Cameras
                             delta + self.street.angle, # Magnetometer
                             alpha, # Encoder on steering wheel
                             v1, # Encoder on motor shaft
                             a1, # Accelerometer
                             (x * cas.cos(self.street.angle) - y * cas.sin(self.street.angle)), # GPS
                             (x * cas.sin(self.street.angle) + y * cas.cos(self.street.angle)), # GPS
                             0*x # Radar
                            )
    
    def compute_truck_scheduling(self):
        # The scheduling is computed in the street reference frame
        vehicles_info = without_keys(self.platoon_status, ["leader", "overtaking"])
        c0s = [s[1] for s in vehicles_info.values()]
        c1s = [s[2] for s in vehicles_info.values()]
        ls = [s[0] for s in vehicles_info.values()]
        S = self.street.length - self.x
        n = len(vehicles_info.keys())

        C = np.zeros((2*n, 2*n))

        for j in range(2*n):
            C[0, j] = 1 if (j % 2 == 0  and j + 2 != 2*n) else 0
            C[0, -1] = -1

        for i in range(n - 1):
            for j in range(2 * n):
                if j == 2*i + 1:
                    C[i + 1,j] = -1
                elif j % 2 == 0 and j != 2*i:
                    C[i + 1,j] = 1

        for i in range(0, n):
            for j in range(2*i, 2*n):
                C[n + i, j : j + 2] = np.array([1,1])
                break

        B = np.zeros((n, 2* n))
        for i in range(0, n):
            for j in range(2*i, 2*n):
                B[i, j : j + 2] = np.array([c0s[i] + c1s[i], c0s[i]])
                break

        b = np.zeros((2*n))
        b[n:] = np.ones(n) * S

        opti = cas.Opti()
        U = opti.variable(n*2)

        life = ls - B @ U

        opti.subject_to( C @ U - b == 0)
        opti.subject_to( U >= np.zeros((n*2,1)))

        # opti.minimize(cas.sumsqr(cas.diff(life) ) )
        opti.minimize(cas.sumsqr(B @ U)) 

        p_opts = {"expand":True}
        s_opts = {"max_iter": 10000}
        opti.solver('ipopt', p_opts, s_opts)

        sol = opti.solve() 

        # Obtain a sequence
        # [S_head, S_in_queue] for each vehicle
        u = np.array(np.split(np.round(sol.value(U),0),n))
        decreasing_order = u[:, 1].argsort()
               
        # Store the scheduling for each vehicle
        # Starting from the actual leading vehicle to the last one
        # The schedule must be shifted in order to consider the actual leading vehicle
        # As the first vehicle executing the leader role of the platoon

        # {v0: [0, S_head, S_in_queue], v1: [S_overtaking, S_head, S_in_queue], ...}
        # For the actual leader there is no need to overtake
        # The others must do it at the right moment
        shift = np.where(decreasing_order == n-1)[0][0]
        order = np.roll(decreasing_order, -shift)
        u = u[order]

        vehicles = np.array([v for v in vehicles_info])
        vehicles = vehicles[order]
        schedule = {}

        s = 0
        for i,v in enumerate(vehicles):
            schedule[v] = [s, u[i]]
            s += u[i][0]

        self.schedule = schedule

    
    def set_schedule(self, schedule):
        self.schedule = schedule
        self.x_received_schedule = self.x

    def set_leader(self):
        self.leader = True
        self.platoon_status["leader"] = self

    def unset_leader(self):
        self.leader = False
        self.platoon_status["leader"] = None


def without_keys(d, keys):
    return {x: d[x] for x in d if x not in keys}

        
        
        

        


        





