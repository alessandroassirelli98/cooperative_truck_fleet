import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from street import Street, Lane
from vehicle import Vehicle
import conf
plt.style.use('seaborn')


street = Street(0, 0, 1000, 0)
lanes = street.lanes
n_vehicles = 2

dt = 0.1
T = 100
N = int(T/dt)

vehicles_list = [Vehicle(street, lanes[0], 0, 10, dt, N+1, L=3)]
# vehicles_list[0].c1 = 10 / 1000
# vehicles_list[0].autonomy = 30
# vehicles_list[0].status = [vehicles_list[0].autonomy , vehicles_list[0].c0, vehicles_list[0].c1, vehicles_list[0].x]
[vehicles_list.append(Vehicle(street, lanes[0], vehicles_list[i].x - 10, 10, dt, N+1, L=3)) for i in range(n_vehicles-1)]


def update_platoon_order(vehicles_list, prev_leader=None):
    platoon_vehicles = []
    for v in vehicles_list:
        platoon_vehicles.append(v)
    platoon_vehicles.sort(key=lambda x: x.s, reverse=True)
    
    if platoon_vehicles[0] != prev_leader:
        platoon_vehicles[0].schedule = platoon_vehicles[1].schedule
        platoon_vehicles[0].xs_schedule = platoon_vehicles[1].xs_schedule
        platoon_vehicles[0].set_leader(last_leader=prev_leader)
        prev_leader = platoon_vehicles[0]
    else:
        platoon_vehicles[0].set_leader()

    for i in range(1, len(platoon_vehicles)):
        platoon_vehicles[i].unset_leader()

    return platoon_vehicles


### Animation of the simulation

def update_animation(frame):
    ax.clear()
    ax.set(xlim=[0, street.x_end], ylim=[lanes[0].y_start, lanes[0].y_end], xlabel='Time [s]', ylabel='Z [m]')
    log_xydelta = [v.log_xydelta_true for v in vehicles_list]
    log_xydelta = np.array(log_xydelta)
    for i  in range(n_vehicles):
        scat[i] = plt.plot(log_xydelta[i, frame, 0], log_xydelta[i, frame, 1],
                           marker = (3, 0, -90 + log_xydelta[i, frame, 2]*180/np.pi), markersize=10, linestyle='None')
    return scat

def distance (v1, v2):
    return np.sqrt((v1.x - v2.x)**2 + (v1.y - v2.y)**2)


times = np.linspace(0, T, N)
freq = 0.36
k_optimize = 100
u_first_vehicle = np.sin(freq* times)
schedule_available = False
if __name__ == '__main__':
    platoon_vehicles = [vehicles_list[0]]
    for t in range(N):
        platoon_vehicles = update_platoon_order(vehicles_list, platoon_vehicles[0])

        # Messaging cycle from leader to last vehicle
        for i, v in enumerate(platoon_vehicles):
            if i != len(platoon_vehicles)-1 and distance(platoon_vehicles[i+1], v) < conf.comm_range:
                if i == 0:
                    platoon_vehicles[0].update_overtaking()
                    schedule_available = False

                platoon_vehicles[i+1].schedule["overtaking"] = platoon_vehicles[i].schedule["overtaking"]
                platoon_vehicles[i+1].schedule["leader"] = platoon_vehicles[i].schedule["leader"]
                platoon_vehicles[i+1].schedule["last_leader"] = platoon_vehicles[i].schedule["last_leader"]
                
        # Messaging cycle from last vehicle to leader
        for i, v in reversed(list(enumerate(platoon_vehicles))):
            if i != 0:
                if distance(platoon_vehicles[i-1], v) < conf.comm_range:
                    platoon_vehicles[i-1].platoon_status.update(platoon_vehicles[i].platoon_status)

        # Update the vehicles actions
        for i, v in enumerate(platoon_vehicles):

            # # Then compute the optimization and share back the message            
            if v.leader and v.schedule["overtaking"] is None and t==1:#t!=0 and t%k_optimize == 0:
                v.compute_truck_scheduling() # Compute optimization
                schedule_available = True

            # Calculathe the trucks visible to the lidar
            # if v.in_overtake:
            for ve in vehicles_list: 
                if ve != v and distance(ve, v) < conf.lidar_range:
                    v.add_visible_truck(ve)
                elif ve != v:
                    v.remove_visible_truck(ve)

            if not v.leader:
                if v.lane == platoon_vehicles[i-1].lane:
                    to_follow = platoon_vehicles[i-1]

                elif i == 1 :
                    v.update()
                    continue

                else:
                    to_follow = platoon_vehicles[i-2]

                talk = True if distance(to_follow, v) < conf.comm_range else False
                v.update(to_follow, talk) 
            else:
                v.update()
      
                
                    
    legend = ["Vehicle {}".format(i) for i in range(n_vehicles)]
    
    for v in vehicles_list:
        plt.figure()
        for i in range(len(v.S)):
            plt.subplot(7,1,i+1)
            plt.title('S[{}]'.format(i))
            plt.plot(v.estimator.S[i,:])
            plt.plot(v.estimator.S_hat[i,:])

        plt.figure()
        for i in range(len(v.S)):
            plt.subplot(7,1,i+1)
            plt.title('P[{}]'.format(i))
            plt.plot(v.estimator.P[i,i,:])
    

    

    plt.figure()
    log_s = [v.log_s for v in vehicles_list]
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
    plt.title('xy_pos_true')
    log_xydelta = [v.log_xydelta_true for v in vehicles_list]
    log_xydelta = np.array(log_xydelta)
    for i  in range(n_vehicles):
        plt.plot(log_xydelta[i, :, 0], log_xydelta[i, :, 1])
    plt.legend(legend)
    street.plot_street()


    plt.figure()
    plt.title('xy_pos_estimation')
    log_xy_hat = [v.estimator.S_hat[:2,:] for v in vehicles_list]
    for i  in range(n_vehicles):
        plt.plot(log_xydelta[i, :, 0], log_xydelta[i, :, 1])
        plt.plot(log_xy_hat[i][0, :], log_xy_hat[i][1, :])
    plt.legend(legend)

    # battery_levels = {"optimized":[], "standard":[]}
    # not_optimal = pd.read_csv("not_optimal_life.out", delimiter=',', header=None).values.flatten()
    # optimal = pd.read_csv("optimal_life.out", delimiter=',', header=None).values.flatten()
    

    # # for i, v in enumerate(vehicles_list):
    # #     battery_levels["optimized"].append(optimal[i])
    # #     battery_levels["standard"].append(not_optimal[i])

    # for i, v in enumerate(vehicles_list):
    #     battery_levels["optimized"].append(v.autonomy)
    #     battery_levels["standard"].append(not_optimal[i])

    # x = np.arange(len(vehicles_list))  # the label locations
    # width = 0.25  # the width of the bars
    # multiplier = 0

    # fig, ax = plt.subplots(layout='constrained')

    # for attribute, measurement in battery_levels.items():
    #     offset = width * multiplier
    #     rects = ax.bar(x + offset, measurement, width, label=attribute)
    #     ax.bar_label(rects, padding=3)
    #     multiplier += 1

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Length (mm)')
    # ax.set_title('Battery usage comparison')
    # ax.set_xticks(x + width, vehicles_list)
    # ax.legend(loc='upper left', ncols=3)


    ### Show the simulation
    if conf.animate:
        scat = []
        fig, ax = plt.subplots()
        ax.set(xlim=[0, street.x_end], ylim=[lanes[0].y_start, lanes[0].y_end], xlabel='Time [s]', ylabel='Z [m]')
        log_xydelta = [v.log_xydelta for v in vehicles_list]
        log_xydelta = np.array(log_xydelta)
        for i  in range(n_vehicles):
            scat.append(plt.plot(log_xydelta[i, 0, 0], log_xydelta[i, 0, 1],
                        marker = (3, 0, log_xydelta[i, 0, 2]*180/np.pi), markersize=20, linestyle='None'))

        ani = animation.FuncAnimation(fig=fig, func=update_animation, frames=N, interval=dt* 500)

    plt.show()
