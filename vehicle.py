from collections import OrderedDict
import numpy as np
import cvxpy as cp
import conf
from street import Street, Lane
from pid import PID
from ppc import PPC
from estimator import Estimator
from surrounding_estimator import SurroundingEstimator
import casadi as cas

class Vehicle:
    def __init__(self, street:Street, lane:Lane, s, v, dt, N, L=3, starting_battery=100):
        self.pid_steer = PID(kp=conf.kp_pos, kd=conf.kd_pos, ki=conf.ki_pos, integral_sat=conf.v_max)
        self.pid_vel = PID(kp=conf.kp_vel, kd=conf.kd_vel, ki=conf.ki_vel, integral_sat=conf.v_max)
        
        
        self.steer_control = PPC()
        self.steer_control.lookAheadDistance = 50

        self.in_overtake = False
        self.leader = False
        self.init = False
        self.cnt = 0

        self.dt = dt
        self.street = street
        self.lane = lane
        self.x, self.y = self.lane.s_to_xy(s)
        self.s = s
        self.delta = 0
        self.alpha = 0
        self.v = v
        self.v_des = v
        self.omega = 0
        self.a = 0
        self.u_fwd = 0
        self.u = np.array([self.u_fwd, self.omega])
        self.path = np.array([[lane.x_start, lane.y_start], [lane.x_end, lane.y_end]])
        
        self.S0 = np.array([self.x, self.y, self.delta, self.alpha, v, self.a, street.angle])
        self.S = self.S0

        self.L = L # Wheelbase of vehicle
        self.tau = conf.tau # Time constant of the acceleration
        self.r = conf.r # parameters that indicates standstill spacing
        self.h_spacing = conf.h

        self.v_max = conf.v_max
        self.a_max = conf.a_max

        
        

        # The nonlinear model (Obtained by Euler discretization)
        self.creta_sys_model()
        self.update_rotation_matrices()

        self.estimator = Estimator(self, dt, N)


        
        
        self.e1 = 0 

        # 100 l
        # 1 l/10 km
        # v = 10 km/h
        # t = 1h
        # Autonomy is given in liters of fuel
        # C0 is the fuel consumption expressed in % of battery per meter
        # C1 is the additional fuel consumption in % per meter if the vehicle is the leader
        self.starting_battery = starting_battery
        self.autonomy = self.starting_battery
        self.c0 = 100/100 / 1000  
        self.c1 = 100/100 / 1000  
        self.status = [self.autonomy, self.c0, self.c1, self.x]
        self.platoon_status = OrderedDict()
        self.platoon_status[self] = self.status
        self.xs_schedule = {}

        self.schedule = OrderedDict()
        self.schedule["leader"] = self
        self.schedule["last_leader"] = self
        self.schedule["overtaking"] = None
        self.xs_schedule = {}

        self.trucks_visible = []

        self.log_u = []
        self.log_u_unc = []
        self.log_s = []
        self.log_x = []
        self.log_y = []
        self.log_v = []
        self.log_e = []
        self.log_path = []
        self.log_xydelta = []
        self.log_xydelta_true = []


    def track_front_vehicle(self, front_vehicle, can_talk = False):

        # If they can talk use the informations from the front vehicle
        if can_talk:
            self.e1 = front_vehicle.x - self.x - self.r - self.h_spacing * self.v
            self.e2 = front_vehicle.v - self.v - self.h_spacing * self.a
            self.e3 = front_vehicle.a - self.a - (1/conf.tau * (- self.a + self.u_fwd) * self.h_spacing)

            self.u_fwd += 1/self.h_spacing * ( - self.u_fwd + 
                                  conf.kp*self.e1 + 
                                  conf.kd*self.e2 + 
                                  conf.kdd*self.e3 + 
                                  front_vehicle.u_fwd) * self.dt
            
        # If they cannot communicate, and the radar sees the front vehicle
        # the follower vehicle takes a measurement
        # elif abs(front_vehicle.S[0] - self.S[0]) < conf.radar_range:
        #     eps_radar = self.estimator.eps[8, self.cnt]
        #     eps_radardot = np.random.normal(0, 2 * conf.sigma_radar / self.dt)
        #     self.e1 = front_vehicle.S[0] - self.S[0] + eps_radar - self.r - self.h_spacing * self.v
        #     self.e2 = front_vehicle.S[4] - self.S[4] + eps_radardot - self.h_spacing * self.a
        #     self.u_fwd += 1/self.h_spacing * ( - self.u_fwd + conf.kp*self.e1 + conf.kd*self.e2) * self.dt

        # If they cannot communicate, and the radar does not see the front vehicle
        # Go with the actual speed
        else: 
            self.v_des = self.v

    def set_desired_velocities(self, v_des):
        self.u_fwd = self.pid_vel.compute(v_des - self.v, self.dt)

    def overtake(self):
            
            last_leader = self.schedule["last_leader"] if self.schedule["leader"] == self else self.schedule["leader"]
            if last_leader is not None:
                v_leader = last_leader.v
                x_leader = last_leader.x

                if self.x < x_leader + last_leader.r + last_leader.h_spacing * v_leader:
                    self.lane = self.street.lanes[1]
                    beta = self.S[6]
                    x_target = self.x + self.street.lane_width
                    y_target = self.street.lane_width/2
                    self.v_des = self.v_max
                    self.path = np.array([[self.x, self.y], 
                                    [x_target, y_target], 
                                    [self.lane.x_end, self.lane.y_end]]) # Change lane
                    self.in_overtake = True
                    self.schedule["overtaking"] = self

                else: # When the vehicle is in front of the leader go back to lane 0
                    self.lane = self.street.lanes[0]
                    x_target = self.x + self.street.lane_width
                    y_target = self.lane.y_end
                    self.path = np.array([[self.x, self.y], 
                                    [x_target, y_target], 
                                    [self.lane.x_end, self.lane.y_end]]) # Change lane

                    self.v_des = 10 
                    self.in_overtake = False
                    self.schedule["overtaking"] = None

                
    def update(self, front_vehicle = None, can_talk = False):
        if self.schedule["overtaking"] == self:
            self.in_overtake = True        
        
        if not self.leader and not self.in_overtake and front_vehicle:
            self.track_front_vehicle(front_vehicle, can_talk)   

        elif self.in_overtake:
            self.overtake()
            self.u_fwd = self.pid_vel.compute(self.v_des - self.v, self.dt)

        else:    
            self.u_fwd = self.pid_vel.compute(self.v_des - self.v, self.dt)
            

        xy_position = np.array([self.x, self.y])
        alpha_des = self.steer_control.ComputeSteeringAngle(self.path, xy_position, self.delta, self.L)
        self.omega = self.pid_steer.compute(alpha_des - self.alpha, self.dt)  # Compute steering angle velocity
        

        self.u = np.array([self.u_fwd, self.omega])

        self.estimator.run_filter(self.u, self.cnt)
        self.update_rotation_matrices()

        self.S = self.f_fun(self.S, self.u, [0,0])
        self.S = self.S.full().flatten() # Convert to numpy array
        self.S[5] = np.clip(self.S[5], -self.a_max, self.a_max) # Clip acceleration


        self.x = self.estimator.S_hat[0, self.cnt + 1]
        self.y = self.estimator.S_hat[1, self.cnt + 1]
        self.s = self.street.xy_to_s(self.x, self.y)
        self.delta = self.estimator.S_hat[2, self.cnt + 1]
        self.alpha = self.estimator.S_hat[3, self.cnt + 1]
        self.v = self.estimator.S_hat[4, self.cnt + 1]
        self.a = self.estimator.S_hat[5, self.cnt + 1]

        consumption = self.c0 + self.c1 if self.leader else self.c0
        self.autonomy -= consumption * np.abs(self.v) * self.dt
        self.status = [self.autonomy, self.c0, self.c1, self.x]
        self.platoon_status[self] = self.status
        
        self.log_u.append(self.u)
        # self.log_u_unc.append(self.u_unc)
        self.log_s.append(self.s)
        self.log_x.append(self.x)
        self.log_y.append(self.y)
        self.log_v.append(self.v)
        self.log_e.append(self.e1)
        self.log_path.append(self.path)
        self.log_xydelta.append([self.x, self.y, self.delta])
        self.log_xydelta_true.append([self.S[0], self.S[1], self.S[2]])
        self.cnt += 1

    def add_visible_truck(self, truck):
        if truck not in self.trucks_visible:
            self.trucks_visible.append(truck)
            self.estimator.update_vehcles_list()
    
    def remove_visible_truck(self, truck):
        if truck in self.trucks_visible:
            self.trucks_visible.remove(truck)
            self.estimator.update_vehcles_list()

    def compute_truck_scheduling(self):
        # The scheduling is computed in the street reference frame
        vehicles_info = without_keys(self.platoon_status, ["last_leader", "leader", "overtaking"])
        if len(self.xs_schedule) != 0 and self.sett == False:
            self.dxs_schedule = {s:s.status[3] - self.xs_schedule[s]  for s in vehicles_info}
            self.sett=True
        self.xs_schedule = {s:s.status[3] for s in vehicles_info}
        c0s = [s[1] for s in vehicles_info.values()]
        c1s = [s[2] for s in vehicles_info.values()]
        ls = [s[0] for s in vehicles_info.values()]
        S = self.street.length - self.x
        if S > 0 and self.schedule["leader"] is not None:
            n = len(vehicles_info.keys())

            A = np.zeros((2*n, 2*n))

            for j in range(2*n):
                A[0, j] = 1 if (j % 2 == 0  and j + 2 != 2*n) else 0
                A[0, -1] = -1

            for i in range(n - 1):
                for j in range(2 * n):
                    if j == 2*i + 1:
                        A[i + 1,j] = -1
                    elif j % 2 == 0 and j != 2*i:
                        A[i + 1,j] = 1

            for i in range(0, n):
                for j in range(2*i, 2*n):
                    A[n + i, j : j + 2] = np.array([1,1])
                    break

            P = np.zeros((2*n, 2* n))
            for i in range(0, n):
                for j in range(2*i, 2*n):
                    P[i, j : j + 2] = np.array([c0s[i] + c1s[i], c0s[i]])
                    break

            b = np.zeros((2*n))
            b[n:] = np.ones(n) * S

            q = np.concatenate([np.array(ls), np.zeros(n)])
            G = np.eye(2*n)*-1
            h = np.zeros(2*n)

            # x = cp.Variable(2*n)
            # obj = cp.Minimize(0.5*cp.sum_squares(cp.max(cp.diff(q - P@x))))
            # constraints = [A@x == b, G@x <= h]
            # prob = cp.Problem(obj, constraints)
            # sol = prob.solve()

            opti = cas.Opti()
            U = opti.variable(n*2)

            life = q - P @ U

            opti.subject_to( A @ U - b == 0)
            opti.subject_to( U >= np.zeros((n*2,1)))

            cost = cas.sumsqr(cas.mmax (cas.diff(life)[:n]))
            opti.minimize(cost)
            # opti.minimize(cas.sumsqr(P @ U)) 

            p_opts = {"expand":True}
            s_opts = {"max_iter": 10000}
            opti.solver('ipopt', p_opts, s_opts)

            try:
                sh_guess = [self.schedule[v][1][0] - self.dxs_schedule[v] for v in self.dxs_schedule] 
                u_guess = []
                for i in range(n):
                    u_guess.append(sh_guess[i])
                    u_guess.append(S-sh_guess[i])
                opti.set_initial(U, u_guess)
            except:
                pass

            sol = opti.solve()

            # Obtain a sequence
            # [S_head, S_in_queue] for each vehicle
            u = np.array(np.split(np.round(sol.value(U),0),n))
            decreasing_order = u[:, 1].argsort()
            not_optimal_u = np.array([S,0])
            not_optimal_u = np.concatenate([not_optimal_u, np.array([0, S] * (n-1))])
            np.savetxt('not_optimal_life.out', q - P @ not_optimal_u, delimiter=',')
            np.savetxt('optimal_life.out', sol.value(life), delimiter=',')

                
            # Store the scheduling for each vehicle
            # Starting from the actual leading vehicle to the last one
            # The schedule must be shifted in order to consider the actual leading vehicle
            # As the first vehicle executing the leader role of the platoon

            # {v0: [None, S_head, S_in_queue], v1: [S_overtaking, S_head, S_in_queue], ...}
            # For the actual leader there is no need to overtake
            # The others must do it at the right moment
            for i, ve in enumerate(vehicles_info):
                if ve == self.schedule["leader"]:
                    target = i
            shift = np.where(decreasing_order == target)[0][0]
            order = np.roll(decreasing_order, -shift)
            u = u[order]

            vehicles = np.array([v for v in vehicles_info])
            vehicles = vehicles[order]
            schedule = OrderedDict()

            s = 0
            for i,v in enumerate(vehicles):
                if i == 0:
                    schedule[v] = [None, u[i]]
                else:
                    schedule[v] = [s, u[i]]
                s += u[i][0]

            self.set_schedule(schedule)
            
        else:
            if S <= 0:
                print("No scheduling needed, the road is finished")
            else:
                print("Cannot compute scheduling, no leader")   
            schedule = {"overtaking": None}
            self.set_schedule(schedule)

    def update_overtaking(self):

        for v in without_keys(self.schedule, ["last_leader", "leader", "overtaking"]):
            if self.schedule[v][0] is not None and self.schedule["overtaking"] is None:
                if  (self.platoon_status[v][3] - self.xs_schedule[v]) > self.schedule[v][0] - 1e-3: 
                    self.schedule["overtaking"] = v
                    del self.schedule[v]
                    break
                
    def creta_sys_model(self):
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
        self.S_sym = cas.SX.sym('S', 7)
        self.Nu_sym = cas.SX.sym('Nu', 2)
        self.U_sym = cas.SX.sym('U', 2)

        x = self.S_sym[0]
        y = self.S_sym[1]
        delta = self.S_sym[2]
        alpha = self.S_sym[3]
        v1 = self.S_sym[4]
        a1 = self.S_sym[5]
        beta = self.S_sym[6]

        self.f = cas.vertcat(cas.cos(delta) * v1,
                        cas.sin(delta) * v1,
                        cas.tan(alpha)/self.L * v1,
                        self.U_sym[1] + self.Nu_sym[1],
                        a1,
                        1/self.tau * (-a1 + (self.U_sym[0] + self.Nu_sym[0])),
                    0) * self.dt + self.S_sym
        
        self.f_fun = cas.Function('f_fun', [self.S_sym, self.U_sym, self.Nu_sym], [self.f])

    def update_rotation_matrices(self):
        self.M01 = np.array([[np.cos(self.S[2]), -np.sin(self.S[2]), self.S[0]],
                    [np.sin(self.S[2]), np.cos(self.S[2]), self.S[1]],
                    [0,0,1]]
                )
        self.M10 = np.array([[np.cos(self.S[2]), np.sin(self.S[2]), - self.S [0] * np.cos(self.S[2]) - self.S[1] * np.sin(self.S[2])],
                    [-np.sin(self.S[2]), np.cos(self.S[2]),- self.S[1] * np.cos(self.S[2]) + self.S[0] * np.sin(self.S[2])],
                     [0,0,1]]
                )
    
        # self.M01_hat = np.array([[np.cos(self.estimator.S_hat[2]), -np.sin(self.estimator.S_hat[2]), self.estimator.S_hat[0]],
        #             [np.sin(self.estimator.S_hat[2]), np.cos(self.estimator.S_hat[2]), self.estimator.S_hat[1]],
        #             [0,0,1]]
        #         )
        # self.M10_hat = np.array([[np.cos(self.estimator.S_hat[2]), np.sin(self.estimator.S_hat[2]), - self.estimator.S_hat [0] * np.cos(self.estimator.S_hat[2]) - self.estimator.S_hat[1] * np.sin(self.estimator.S_hat[2])],
        #             [-np.sin(self.estimator.S_hat[2]), np.cos(self.estimator.S_hat[2]),- self.estimator.S_hat[1] * np.cos(self.estimator.S_hat[2]) + self.estimator.S_hat[0] * np.sin(self.estimator.S_hat[2])],
        #              [0,0,1]]
        #         )
        
    def set_schedule(self, schedule):
        self.schedule.update(schedule)

    def set_leader(self, last_leader = None):
        self.leader = True
        self.schedule["leader"] = self
        if last_leader is not None:
            self.schedule["last_leader"] = last_leader

    def unset_leader(self):
        self.leader = False
        self.schedule = OrderedDict()
        self.schedule["leader"] = None
        self.schedule["last_leader"] = None
        self.schedule["overtaking"] = None

def without_keys(d, keys):
    return {x: d[x] for x in d if x not in keys}

        
        
        

        


        





