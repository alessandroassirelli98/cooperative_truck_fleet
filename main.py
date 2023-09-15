import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from street import Street, Lane
from vehicle import Vehicle
import conf
plt.style.use('seaborn')

street = Street(0, 0, 1000, 0)
lanes = street.lanes
n_vehicles = 1

vehicles_list = [Vehicle(street, lanes[0], 10, 00, L=3)]
[vehicles_list.append(Vehicle(street, lanes[0], vehicles_list[i].s - conf.r/2, 0, L=3)) for i in range(n_vehicles-1)]


dt = 0.05
T = 50
N = int(T/dt)
times = np.linspace(0, T, N)
freq = 0.36
u_first_vehicle = np.sin(freq* times)

def update_platoon_order(vehicles_list):
    platoon_vehicles = {}
    for l in lanes:
        platoon_vehicles[l] = []
        for v in vehicles_list:
            if v.lane == l:
                platoon_vehicles[l].append(v)
        platoon_vehicles[l].sort(key=lambda x: x.s, reverse=True)
        if platoon_vehicles[l] != [] : platoon_vehicles[l][0].leader = True 
        return platoon_vehicles


if __name__ == '__main__':
    plt.figure()
    for t in range(N):
        platoon_vehicles = update_platoon_order(vehicles_list)
        for l in platoon_vehicles.keys():
            for i, v in enumerate(platoon_vehicles[l]):
                if i == 0:
                    v.u = u_first_vehicle[t]
                
                if t>20:
                   v.change_lane(lanes[1])
                    # if t < int(N/8):
                    #     # v.set_desired_velocities(10, 0, dt)
                    #     v.u = 2
                    # elif t <= int(N* 2/8):
                    #     v.u = -2
                    # elif t <= int(N* 3/8):
                    #     v.u = 2
                    # elif t <= int(N* 4/8):
                    #     v.u = -2
                    # elif t <= int(N* 5/8):
                    #     v.u = 2
                        # v.set_desired_velocities(0, 0, dt)
                    # else:
                    #     v.set_desired_velocities(10, 0, dt)                    
                    
                else:
                    v.track_front_vehicle(platoon_vehicles[l][i-1], dt, use_velocity_info = True) # follow vehicle in front
                v.update(dt)
                if conf.animate: plt.plot(v.x, v.y, 'bo')
                    
        if conf.animate:
            plt.xlim(street.x_start, 300)
            plt.ylim(street.y_start - street.lane_width, street.y_end + street.lane_width)  
            plt.show(block=False)
            plt.pause(dt)
            # plt.savefig('animation_plot/{}.png'.format(t))
            plt.clf()




    legend = ["Vehicle {}".format(i) for i in range(n_vehicles)]

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
    

    plt.show()