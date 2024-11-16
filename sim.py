import numpy as np
from systems import BicycleModel
from trajectory import compute_circle_start_on_circle, wrap_circular_value
import matplotlib.pyplot as plt
from controller import ctrl_linear

def simulation(params, sim_params):
    N = sim_params['N']
    DT = sim_params['DT']
    K = sim_params['K']
    # integral = sim_params['integral']


    # Set the model.
    car = BicycleModel(params)

    # Des. Traj.
    des_traj_array = np.empty((N, 6))
    theta_d_previous = 0.0
    e_perp_sum = 0.0

    # Outputs dict.
    output_dict_list = []

    action_list = np.empty((N, 2))

    angle = 0.0
    state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    state0 = state.copy()
    state_array = np.empty((N, 6))
    e_integral = np.zeros(4)


    for i in range(N):
        # Compute desired trajectory.
        x_d_I, y_d_I, _, vx_d_I, vy_d_I, _, angle = compute_circle_start_on_circle(angle=angle,
                                                                                dt=DT,
                                                                                v_desired=2.0,
                                                                                initial_state_I=[state0[0], state0[1], state0[2]])
        theta_d = np.arctan2(vy_d_I, vx_d_I)
        omega_d = wrap_circular_value((theta_d - theta_d_previous)/DT)
        theta_d_previous = theta_d
        state_d = np.array([x_d_I, y_d_I, theta_d, vx_d_I, vy_d_I, omega_d])

        action, outputs, e_integral = ctrl_linear(state=state, state_d=state_d, K=K, sim_params=sim_params, e_integral=e_integral)

        # Propagate.
        next_state = car.dynamics(state, action)
        state = next_state.copy()
        
        # Save data.
        des_traj_array[i, :] = state_d
        state_array[i, :] = state
        output_dict_list.append(outputs)
        action_list[i, :] = action
    
    return state_array, des_traj_array, output_dict_list, action_list
        
def plot_system(state_array, des_traj_array, output_dict_list, action_list):
    fig, ax = plt.subplots(2, 2, figsize=(4, 4))
    plt.suptitle('Desired Trajectory')
    ax[0, 0].plot(des_traj_array[:, 0], des_traj_array[:, 1], 'r-', label='xy des')
    ax[0, 0].plot(des_traj_array[0, 0], des_traj_array[0, 1], 'ro')
    ax[0, 0].plot(des_traj_array[-1, 0], des_traj_array[-1, 1], 'rx')

    ax[0, 1].plot(des_traj_array[:, 2], 'g-', label='theta des')

    ax[1, 0].plot(des_traj_array[:, 3], 'r-', label='v_x des (Inertial)')
    ax[1, 0].plot(des_traj_array[:, 4], 'b-', label='v_y des (Inertial)')

    ax[1 ,1].plot(des_traj_array[:, 5], 'g-', label='omega des')
    for i in range(2):
        for j in range(2):
            ax[i, j].legend(loc='upper right', fontsize=8)
    plt.tight_layout()

    plt.figure(figsize=(3, 3))
    plt.plot(des_traj_array[:, 0], des_traj_array[:, 1])
    plt.plot(des_traj_array[0, 0], des_traj_array[0, 1], 'ro')
    plt.plot(state_array[:, 0], state_array[:, 1])
    plt.axis('equal')
    plt.title('Performance')

    e_perp_array = np.array([output_dict['e_perp'] for output_dict in output_dict_list])
    e_perp_dot_array = np.array([output_dict['e_perp_dot'] for output_dict in output_dict_list])
    theta_err_array = np.array([output_dict['theta_err'] for output_dict in output_dict_list])
    omega_err_array = np.array([output_dict['omega_err'] for output_dict in output_dict_list])

    fig, ax = plt.subplots(1, 4, figsize=(10, 2))
    ax[0].plot(e_perp_array, label='e_perp')
    ax[0].legend()
    ax[1].plot(e_perp_dot_array, label='e_perp_dot')
    ax[1].legend()
    ax[2].plot(theta_err_array, label='theta_err')
    ax[2].legend()
    ax[3].plot(omega_err_array, label='omega_err')
    ax[3].legend()
    plt.tight_layout()

    # Velocities.
    v_d_x = np.array([output_dict['v_d_B_x'] for output_dict in output_dict_list])
    v_d_y = np.array([output_dict['v_d_B_y'] for output_dict in output_dict_list])
    plt.figure(figsize=(4, 4))
    plt.plot(state_array[:, 3], label='v_x')
    plt.plot(v_d_x, label='v_x_d')
    plt.plot(state_array[:, 4], label='v_y')
    plt.plot(v_d_y, label='v_y_d')
    plt.legend()

    # Velocities.
    plt.figure(figsize=(5, 5))
    plt.plot(action_list[:, 0], label='u_v')
    # plt.plot(v_d_x, label='u_v')
    plt.plot(action_list[:, 1], label='u_steering')
    # plt.plot(v_d_y, label='u_steering')
    plt.legend()

    plt.show()