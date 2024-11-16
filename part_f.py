import numpy as np
from part_c import calc_gains
from sim import simulation, plot_system


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

    sim_params = {
        'N': 3000,
        'MAX_VEL': 10.0,
        'MAX_STEERING': np.deg2rad(20),
        'K':K,
        'DT': 0.01,
        'L': 0.4,
        'integral': False,
        'feedforward':True,
    }

    state_array, des_traj_array, output_dict_list, action_list = simulation(params=params, sim_params=sim_params)
    plot_system(state_array, des_traj_array, output_dict_list, action_list)