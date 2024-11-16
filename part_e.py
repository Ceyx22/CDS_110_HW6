import numpy as np
from control import lqr
from sim import simulation, plot_system
from part_c import calc_gains

def aug_K(params):
    
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

    n = A.shape[0]  
    print(n)
    A_aug = np.block([
        [A, np.zeros((n, n))],
        [np.eye(n), np.zeros((n, n))]
    ])

    # print(A_aug)
    B_aug = np.vstack((B, np.zeros((n, 1))))
    

    Q_e = np.diag([20, 1, 20, 1])  
    Q_int = np.diag([20, 1, 20, 1])
    Q_aug = np.block([
        [Q_e, np.zeros((n, n))],
        [np.zeros((n, n)), Q_int]
    ])

    R = np.array([[1]])

    
    K_aug, S_aug, E_aug = lqr(A_aug, B_aug, Q_aug, R)
    # print(K_aug)
    return K_aug

if __name__ == "__main__":
    # Model params.
    params = {"dt": 0.01,
                "tau": 0.1,
                'L': 0.4,
                'Cy': 100,
                'mass': 11.5,
                'Iz': 0.5,
    }
    K = calc_gains(params)

    # K = aug_K(params=params)
    # print(K)

    sim_params = {
        'N': 3000,
        'MAX_VEL': 10.0,
        'MAX_STEERING': np.deg2rad(20),
        'K':K,
        'DT': 0.01,
        'L': 0.4,
        'integral': True,
        'feedforward':False,
    }

    state_array, des_traj_array, output_dict_list, action_list = simulation(params=params, sim_params=sim_params)
    plot_system(state_array, des_traj_array, output_dict_list, action_list)


