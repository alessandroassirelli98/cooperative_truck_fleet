import numpy as np

animate = True

kp_pos = 5
kd_pos = 0.5
ki_pos = 0

kp_vel = 1
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

comm_range = 100000
radar_range = 50000
lidar_range = 50

P_gps = 0.1
P_radar = 0.1

### Measurement uncertainties
sigma_y = 1e-2 * 1e-1
sigma_delta = 1e-2 * 1e-1
sigma_alpha = 1e-2 * 1e-1
sigma_v = 1e-1 * 1e-1
sigma_a = 1e-1 * 1e-1
sigma_mag = 1e-2 * 1e-1
sigma_beta = 1e-2 * 1e-1
sigma_x_gps = 1e-1 * 1e-1
sigma_y_gps = 1e-1 * 1e-1
sigma_radar = 1e-1 * 1e-1
sigma_lidar_rho = 1e-1 * 1e-1
sigma_lidar_phi = 1e-1 * 1e-1

sigma_u = 1e0
sigma_omega = 1e-1

PATH_TOL = 10