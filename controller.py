import numpy as np
from trajectory import wrap_circular_value


def ctrl_linear(state:np.ndarray, state_d:np.ndarray, K:np.ndarray, sim_params, integral=False, e_integral=np.zeros(4)) -> np.ndarray:
    MAX_VEL = sim_params['MAX_VEL']
    MAX_STEERING = sim_params['MAX_STEERING']
    DT = sim_params['DT']

    p_I_x, p_I_y, theta, v_B_x, v_B_y, omega = state
    p_d_I_x, p_d_I_y, theta_d, v_d_I_x, v_d_I_y, omega_d = state_d

    p_I = np.array([p_I_x, p_I_y]) # Positions in the inertial frame.
    p_d_I = np.array([p_d_I_x, p_d_I_y]) # Desired positions in the inertial frame.
    v_d_I = np.array([v_d_I_x, v_d_I_y]) # Desired velocities in the inertial frame.

    p_err_I = p_I - p_d_I # Position error in the inertial frame.
    R_d = np.array([[np.cos(theta_d), -np.sin(theta_d)],
                    [np.sin(theta_d), np.cos(theta_d)]])
    R = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    perr_B = R.T @ p_err_I
    perr_d_B = R_d.T @ p_err_I
    v_d_B = R_d.T @ v_d_I

    # e1.
    e_perp = perr_d_B[1]
    # e1_dot.
    e_perp_dot = v_B_y + v_B_x * wrap_circular_value(theta - theta_d)
    # e2.
    theta_err = wrap_circular_value(theta - theta_d)
    # e2_dot.
    omega_err = omega - omega_d

    ###
    # Add the u_steering calculation here as a feedback on e_perp, e_perp_dot, theta_err, omega_err.
    # Do not forget to clip the steering angle between u_steering_min and u_steering_max.
    # Do not forget to clamp the integral gain for adaptation.
    e = np.array([e_perp, e_perp_dot, theta_err, omega_err])
    
    
    if integral:
        e_integral = e_integral + e * DT
        # print(e_integral)
        e_integral = np.clip(e_integral, -MAX_STEERING, MAX_STEERING)

        e = np.concatenate((e, e_integral))
        # print(e_integral)


    u_steering = - K @ e
    # print(u_steering)
    u_steering = np.clip(u_steering[0], -MAX_STEERING, MAX_STEERING)

    ###
    # Add the u_v calculation here from Problem Set 5
    # Do not forget to clip u_v.
    # Using u_v controller from solutions
    K_v = 1.0
    a = -1/0.1
    b = 1/0.1
    u_v = -1/b*(K_v * (v_B_x - v_d_B[0]) + a * v_d_B[0])
    u_v = np.clip(u_v, -MAX_VEL, MAX_VEL)
    
    # Debug params.
    outputs = {
        'e_perp': e_perp,
        'e_perp_dot': e_perp_dot,
        'theta_err': theta_err,
        'omega_err': omega_err,
        'v_d_B_x': v_d_B[0],
        'v_d_B_y': v_d_B[1],
    }

    return np.array([u_v, u_steering]), outputs, e_integral
