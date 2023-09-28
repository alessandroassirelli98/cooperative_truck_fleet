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

### Measurement uncertainties
sigma_y = 1e-1
sigma_delta = 1e-1
sigma_alpha = 1e-2
sigma_v = 1e-1
sigma_a = 1e-1
sigma_mag = 1e-1
sigma_beta = 1e-2
sigma_x_gps = 1e-1
sigma_y_gps = 1e-1
sigma_radar = 1e-1


Q = np.asarray(np.matrix([[1, 0], [0, 1]])) * 1e0