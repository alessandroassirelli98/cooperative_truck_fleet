import numpy as np

animate = False

kp_pos = 5
kd_pos = 0.5
ki_pos = 0

kp_vel = 5
kd_vel = 0.2
ki_vel = 0

# String stability analysis
kp = 0.2
kd = 0.7
kdd = 0

tau = 0.1

h = 0.7
r = 30

v_max = 40
a_max = 60

sigma_street_angle = 1e-5
Q = np.asarray(np.matrix([[1, 0], [0, 1]])) * 1e-2
R = np.eye(6) * 1e-2


# Generate H matrix according to the available sensors
H_ = [np.matrix([[1,0,0,0,0,0], [1,0,0,0,0,0]]), # H_GPS
    np.matrix([[0,0,1,0,0,0]]), # H_MAGNETOMETER
    np.matrix([[0,0,0,1,0,0]]), # H_STEER_ENCODER
    np.matrix([[0,0,0,0,1,0]]), # H_WHEEL_ENCODER
    np.matrix([[0,0,0,0,0,1]]) # H_IMU
]

H = np.asarray(np.concatenate(H_, axis=0))

R = R**2
sigma_u = 1e0