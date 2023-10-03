import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from street import Street, Lane
from vehicle import Vehicle
import conf
plt.style.use('seaborn')


street = Street(0, 0, 5000, 100)
lanes = street.lanes
n_vehicles = 5

dt = 0.05
T = 150
N = int(T/dt)

vehicles_list = [Vehicle(street, lanes[0], 20, 20, dt, N+1, L=3)]
[vehicles_list.append(Vehicle(street, lanes[0], vehicles_list[i].x - conf.r/2, 0, dt, N+1, L=3)) for i in range(n_vehicles-1)]


def update_platoon_order(vehicles_list):
    platoon_vehicles = {}

    for l in lanes:
        platoon_vehicles[l] = []
        for v in vehicles_list:
            if v.lane == l:
                platoon_vehicles[l].append(v)
        platoon_vehicles[l].sort(key=lambda x: x.x, reverse=True)
        if platoon_vehicles[l] != []:
            platoon_vehicles[l][0].set_leader()
            for v in platoon_vehicles[l][1:]: v.unset_leader()

    return platoon_vehicles


### Animation of the simulation

def update_animation(frame):
    ax.clear()
    ax.set(xlim=[0, street.x_end], ylim=[-street.lane_width, street.lane_width], xlabel='Time [s]', ylabel='Z [m]')
    log_xydelta = [v.log_xydelta for v in vehicles_list]
    log_xydelta = np.array(log_xydelta)
    for i  in range(n_vehicles):
        scat[i] = plt.plot(log_xydelta[i, frame, 0], log_xydelta[i, frame, 1],
                           marker = (3, 0, -90 + log_xydelta[i, frame, 2]*180/np.pi), markersize=10, linestyle='None')
    return scat


times = np.linspace(0, T, N)
freq = 0.36
k_optimize = 10000
u_first_vehicle = np.sin(freq* times)

if __name__ == '__main__':
    for t in range(N):
        platoon_vehicles = update_platoon_order(vehicles_list)
        for l in platoon_vehicles.keys():
            for i, v in enumerate(platoon_vehicles[l]):

                if abs(platoon_vehicles[l][i-1].x - v.x) < conf.comm_range and i != 0:
                    v.platoon_status[platoon_vehicles[l][i]] = v.status
                    v.platoon_status.update(platoon_vehicles[l][i - 1].platoon_status)
                else:
                    # If not in communication range, it knows only its status
                    v.platoon_status[platoon_vehicles[l][i]] = v.status

                if v == platoon_vehicles[l][-1] and t % k_optimize == 0:
                    # compute optimization
                    v.compute_truck_scheduling()
                    # Messaging
                    for j in reversed(range(len(platoon_vehicles[l]))):
                        if abs(platoon_vehicles[l][j-1].x - platoon_vehicles[l][j].x) < conf.comm_range:
                            platoon_vehicles[l][j-1].set_schedule(platoon_vehicles[l][j].schedule)

                if i != 0:
                    v.update(platoon_vehicles[l][i-1]) # follow vehicle in front
                else:
                    v.update()

                
                
                
                
                    
    legend = ["Vehicle {}".format(i) for i in range(n_vehicles)]

    

    plt.figure()
    log_s = [v.log_x for v in vehicles_list]
    log_s = np.array(log_s)
    plt.title('s')
    for i  in range(n_vehicles):
        plt.plot(times, log_s[i])
    plt.legend(legend)


    
    plt.figure()
    plt.title('v')
    log_v = [v.log_v for v in vehicles_list]
    log_v = np.array(log_v)
    for i  in range(n_vehicles):
        plt.plot(times, log_v[i])
    plt.legend(legend)
    


    plt.figure()
    plt.title('u')
    log_u = [v.log_u for v in vehicles_list]
    log_u = np.array(log_u)
    for i  in range(n_vehicles):
        plt.plot(times, log_u[i])
    plt.legend(legend)
    


    plt.figure()
    plt.title('error')
    log_e = [v.log_e for v in vehicles_list]
    log_e = np.array(log_e)
    for i  in range(n_vehicles):
        plt.plot(times, log_e[i])
    plt.legend(legend)
    

    plt.figure()
    plt.title('xy_pos_world')
    log_xydelta = [v.log_xydelta_world for v in vehicles_list]
    log_xydelta = np.array(log_xydelta)
    for i  in range(n_vehicles):
        plt.plot(log_xydelta[i, :, 0], log_xydelta[i, :, 1])
    plt.legend(legend)

    plt.figure()
    plt.title('xy_pos')
    log_xydelta = [v.log_xydelta for v in vehicles_list]
    log_xydelta = np.array(log_xydelta)
    for i  in range(n_vehicles):
        plt.plot(log_xydelta[i, :, 0], log_xydelta[i, :, 1])
    plt.legend(legend)

    plt.figure()
    plt.title('xy_pos_estimation')
    log_xy_hat = [v.estimator.S_hat[:2,:] for v in vehicles_list]
    for i  in range(n_vehicles):
        plt.plot(log_xydelta[i, :, 0], log_xydelta[i, :, 1])
        plt.plot(log_xy_hat[i][0, :], log_xy_hat[i][1, :])
    plt.legend(legend)


    ### Show the simulation
    if conf.animate:
        scat = []
        fig, ax = plt.subplots()
        ax.set(xlim=[0, street.x_end], ylim=[-street.lane_width, street.lane_width], xlabel='Time [s]', ylabel='Z [m]')
        log_xydelta = [v.log_xydelta for v in vehicles_list]
        log_xydelta = np.array(log_xydelta)
        for i  in range(n_vehicles):
            scat.append(plt.plot(log_xydelta[i, 0, 0], log_xydelta[i, 0, 1],
                        marker = (3, 0, log_xydelta[i, 0, 2]*180/np.pi), markersize=20, linestyle='None'))

        ani = animation.FuncAnimation(fig=fig, func=update_animation, frames=N, interval=dt* 500)

    plt.show()