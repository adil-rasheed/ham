"""
inspect_dataset.py

Written by Sindre Stenen Blakseth, 2021.

Script for inspecting datasets used for training and evaluating ML correction models.
"""

########################################################################################################################
# Package imports.

import joblib
import matplotlib.pyplot as plt
import numpy as np
import numpy_indexed as npi
import os

########################################################################################################################
# File imports.

import config
import physics

########################################################################################################################

def main():
    if config.system in [1,2]:
        indices = [0, 100, 200, 400, 800, 1600, 4000]
    elif config.system == 3:
        indices = [0, 100, 200, 400, 800, 1500, 3000]
    elif config.system == 4:
        indices = [0,9,19,29,39,49,59,69,79,89, 99]
    elif config.system == 5:
        indices = [0, 10, 100, 1000]
    else:
        raise Exception("Invalid system selection.")
    save_filepath = os.path.join(config.datasets_dir, config.data_tag + ".sav")
    if os.path.exists(save_filepath):
        simulation_data = joblib.load(save_filepath)
    else:
        raise Exception("No matching dataset found.")

    plt.figure()
    plt.title("Uncorrected")
    for index in indices:
        plt.plot(simulation_data['x'], simulation_data['unc'][index], label=index)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(config.results_dir, config.data_tag + '/unc.pdf'), bbox_inches='tight')

    plt.figure()
    plt.title("Reference")
    for index in indices:
        plt.plot(simulation_data['x'], simulation_data['ref'][index], label=index)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(config.results_dir, config.data_tag + '/ref.pdf'), bbox_inches='tight')

    plt.figure()
    plt.title("Source")
    for index in indices:
        plt.plot(simulation_data['x'][1:-1], simulation_data['src'][index], label=index)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(config.results_dir, config.data_tag + '/src.pdf'), bbox_inches='tight')

    if config.exact_solution_available:
        plt.figure()
        plt.title("Correction source term vs source error")
        plt.plot(simulation_data['x'][1:-1], simulation_data['src'][-1], label='ML Target')
        x_dense = np.linspace(simulation_data['x'][0], simulation_data['x'][-1], 1001, endpoint=True)
        plt.plot(x_dense,
                 config.get_q_hat(x_dense, config.t_end) - config.get_q_hat_approx(x_dense, config.t_end),
                 label='Real - approx')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(config.results_dir, config.data_tag + '/src2.pdf'), bbox_inches='tight')


    if config.exact_solution_available:
        plt.figure()
        plt.title("Exact vs ref")
        plt.plot(simulation_data['x'], simulation_data['ref'][-1], '.', label="ref")
        plt.plot(simulation_data['x'], config.get_T_exact(config.nodes_coarse, config.t_end), label="exact")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(config.results_dir, config.data_tag + '/exact.pdf'), bbox_inches='tight')

    T0 = simulation_data['unc'][0]
    t = 0
    index = 0
    T_old = T0
    T_old_ = T0
    errors = []
    errors_ = []
    while t < config.t_end:
        t_new = np.around(t + config.dt_coarse, decimals=10)
        T_new = physics.simulate(
            config.nodes_coarse, config.faces_coarse,
            T_old, config.get_T_a, config.get_T_b,
            config.get_k_approx, config.get_cV, config.rho, config.A,
            config.get_q_hat_approx, simulation_data['src'][index+1],
            config.dt_coarse, t, t_new, False
        )
        T_new_ = physics.simulate(
            config.nodes_coarse, config.faces_coarse,
            T_old_, config.get_T_a, config.get_T_b,
            config.get_k_approx, config.get_cV, config.rho, config.A,
            config.get_q_hat_approx,
            config.get_q_hat(config.nodes_coarse[1:-1],t_new)-config.get_q_hat_approx(config.nodes_coarse[1:-1],t_new),
            config.dt_coarse, t, t_new, False
        )
        index += 1
        t = np.around(t + config.dt_coarse, decimals=10)
        if index < simulation_data['ref'].shape[0]:
            errors.append(np.sqrt(np.sum((T_new - simulation_data['ref'][index])**2)))
            errors_.append(np.sqrt(np.sum((T_new_ - simulation_data['ref'][index]) ** 2)))
        T_old = T_new
        T_old_ = T_new_

    print("Corrected:", T_old)
    print("Reference:", simulation_data['ref'][-1])
    print("End diff:", T_old - simulation_data['ref'][-1])

    plt.figure()
    plt.title("Errors")
    plt.plot(np.arange(len(errors)), errors, label="ML Target")
    plt.plot(np.arange(len(errors_)), errors_, label="Real - approx")
    plt.plot()
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(config.results_dir, config.data_tag + '/final_errors.pdf'), bbox_inches='tight')

    plt.figure()
    plt.title("Final profiles")
    plt.plot(config.nodes_coarse, T_old, '.', label='ML Target')
    plt.plot(config.nodes_coarse, T_old_, '.', label="Real - approx")
    x_dense = np.linspace(config.x_a, config.x_b, 1001, endpoint=True)
    plt.plot(x_dense, config.get_T_exact(x_dense, config.t_end), label="Exact")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(config.results_dir, config.data_tag + '/final_profiles.pdf'), bbox_inches='tight')

########################################################################################################################'

if __name__ == "__main__":
    main()

########################################################################################################################