# Problem 1
# Part D
import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.linalg import inv
from sim import simulation, plot_system


def calc_lqr(params):
    
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

    B = np.array([[0], [C_y/m], [0], [C_y*L/(2*I_z)]])

    # Q = np.diag([1, 15, 1, 15])
    Q = np.diag([15, 1, 15, 1])
    R = np.array([[1]])

    P = solve_continuous_are(A, B, Q, R)
    K_lqr = inv(R) @ B.T @ P
    return K_lqr

if __name__ == "__main__":
    # Model params.
    params = {"dt": 0.01,
                "tau": 0.1,
                'L': 0.4,
                'Cy': 100,
                'mass': 11.5,
                'Iz': 0.5,
    }


    K = calc_lqr(params=params)

    sim_params = {
        'N': 3000,
        'MAX_VEL': 10.0,
        'MAX_STEERING': np.deg2rad(20),
        'K':K,
        'DT': 0.01,
        'L': 0.4,
        'integral': False,
        'feedforward':False,
    }

    state_array, des_traj_array, output_dict_list, action_list = simulation(params=params, sim_params=sim_params)
    plot_system(state_array, des_traj_array, output_dict_list, action_list)

