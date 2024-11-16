# Problem 1
# Part C
import numpy as np
import scipy
from sim import simulation, plot_system

def calc_gains(params):
    v_xb = 2.0
    L = params['L']
    m = params['mass']
    C_y = params['Cy']
    I_z = params['Iz']

    A = np.array([
        [0, 1, 0, 0],
        [0, -C_y/(m*v_xb), C_y/m, 0],
        [0, 0, 0, 1],
        [0, 0, 0, (-L**2 * C_y)/(2*I_z*v_xb)]
    ])

    B = np.array([[0], [C_y/m], [0], [(C_y*L)/(2*I_z)]])

    des_poles = np.array([-5.0, -5.5, -6.0, -6.5])
    place_obj = scipy.signal.place_poles(A, B, des_poles)
    K = place_obj.gain_matrix 
    return K

if __name__ == "__main__":
    # Model params.
    params = {"dt": 0.01,
                "tau": 0.1,
                'L': 0.4,
                'Cy': 100,
                'mass': 11.5,
                'Iz': 0.5,
    }

    K = calc_gains(params=params)
    sim_params = {
        'N': 3000,
        'MAX_VEL': 10.0,
        'MAX_STEERING': np.deg2rad(20),
        'K':K, 
        'DT': 0.01,
        'integral': False
    }

    state_array, des_traj_array, output_dict_list, action_list = simulation(params=params, sim_params=sim_params)
    plot_system(state_array, des_traj_array, output_dict_list, action_list)