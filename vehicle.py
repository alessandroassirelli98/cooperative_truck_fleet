import numpy as np
import conf
from street import Street, Lane
from pid import PID

class Vehicle:
    def __init__(self, street:Street, lane:Lane, s, v, L=3):
        self.pid_pos = PID(kp=conf.kp_pos, kd=conf.kd_pos, ki=conf.ki_pos, integral_sat=conf.v_max)
        self.pid_vel = PID(kp=conf.kp_vel, kd=conf.kd_vel, ki=conf.ki_vel, integral_sat=conf.v_max)
        self.path = np.array([[lane.x_start, lane.y_start], [lane.x_end, lane.y_end]])

        self.street = street
        self.lane = lane
        self.s = s
        self.x, self.y = lane.s_to_xy(s)
        self.delta = 0
        self.alpha = 0

        self.L = L
        self.r = conf.r # parameters that indicates standstill spacing
        self.h = conf.h # parameters that indicates spacing between front vehicles d  = h * v

        self.v_max = conf.v_max
        self.a_max = conf.a_max

        self.q = np.array([self.x, self.y, self.delta, self.alpha])
        self.dq = np.array([np.cos(self.delta) * v,
                            np.sin(self.delta) * v, 
                            np.tan(self.alpha)/L * v, 
                            0])
        self.state = np.array([self.x, self.y, self.delta, self.alpha, v]) 
        self.v = v
        self.a = 0
        self.u = 0
        self.e1 = 0 
        self.omega = 0

        self.radar_range = 100
        self.leader = False

        self.log_u = []
        self.log_s = []
        self.log_v = []
        self.log_e = []
        self.log_path = []


    def track_front_vehicle(self, front_vehicle, dt, use_velocity_info = True):
        self.e1 = front_vehicle.s - self.s - self.r - self.h * self.v
        self.e2 = front_vehicle.v - self.v - self.h * self.a
        self.e3 = front_vehicle.a - self.a - (1/conf.tau * (- self.a + self.u) * self.h)

        if use_velocity_info:
            self.u += 1/self.h * ( - self.u + conf.kp*self.e1 + conf.kd*self.e2 + conf.kdd*self.e3 + front_vehicle.u) * dt
        else:
            self.u += 1/self.h * ( - self.u + conf.kp*self.e1 + conf.kd*self.e2 + conf.kdd*self.e3) * dt
        self.omega = 0
    
    # def set_desired_velocities(self, v_des, omega_des, dt):
    #     self.u = self.pid_vel.compute(v_des - self.v, dt)
    #     self.omega = self.pid_vel.compute(omega_des - self.omega, dt)

    def change_lane(self, lane: Lane):
        x_target = self.x + 1/np.tan(np.pi/4) + self.street.lane_width
        y_target = lane.y_start
        self.path = np.array([[self.x, self.y], [x_target, y_target], [lane.x_end, lane.y_end]])



    def update(self, dt):

        self.a = self.a + 1/conf.tau * (- self.a + self.u) *dt
        self.a = np.clip(self.a, -self.a_max, self.a_max)

        self.v = self.v + self.a * dt

        self.dq = np.array([np.cos(self.q[2]) * self.v,
                            np.sin(self.q[2]) * self.v, 
                            np.tan(self.q[3])/self.L * self.v, 
                            self.omega])
        
        self.q = self.q + self.dq * dt
        self.x = self.q[0]
        self.y = self.q[1]
        self.delta = self.q[2]
        self.alpha = self.q[3]

        self.s = self.street.xy_to_s(self.x, self.y)
        self.log_u.append(self.u)
        self.log_s.append(self.s)
        self.log_v.append(self.v)
        self.log_e.append(self.e1)
        self.log_path.append(self.path)