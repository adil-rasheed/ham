"""
inspect_systems.py

Written by Sindre Stenen Blakseth, 2021.

Script for inspecting the temporal behaviour of different physical systems.
"""

########################################################################################################################
# Package imports.

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
    indices = [0, 100, 200, 400, 800, 1600, 4000]

    # Perform coarse-scale simulation with constant k.
    unc_Ts_constk = np.zeros((config.Nt_coarse, config.N_coarse + 2))
    unc_Ts_constk[0] = config.get_T0(config.nodes_coarse)
    for i in range(1, config.Nt_coarse):
        unc_Ts_constk[i] = physics.simulate(
            config.nodes_coarse, config.faces_coarse,
            unc_Ts_constk[i - 1], config.T_a, config.T_b,
            lambda x: np.ones_like(x) * config.k_ref, config.get_cV, config.rho, config.A,
            config.get_q_hat, np.zeros_like(config.nodes_coarse[1:-1]),
            config.dt_coarse, config.dt_coarse*(i-1), config.dt_coarse*i, False
        )

    # Perform coarse-scale simulation.
    unc_Ts = np.zeros((config.Nt_coarse, config.N_coarse + 2))
    unc_Ts[0] = config.get_T0(config.nodes_coarse)
    for i in range(1, config.Nt_coarse):
        unc_Ts[i] = physics.simulate(
            config.nodes_coarse, config.faces_coarse,
            unc_Ts[i - 1], config.T_a, config.T_b,
            config.get_k, config.get_cV, config.rho, config.A,
            config.get_q_hat, np.zeros_like(config.nodes_coarse[1:-1]),
            config.dt_coarse, config.dt_coarse*(i-1), config.dt_coarse*i, False
        )

    plt.figure()
    for index in indices:
        plt.plot(config.nodes_coarse, unc_Ts[index], label=index)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(config.results_dir, 'debug_t/unc.pdf'), bbox_inches='tight')

    plt.figure()
    for index in indices:
        plt.plot(config.nodes_coarse, unc_Ts_constk[index], label=index)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(config.results_dir, 'debug_t/unc_const.pdf'), bbox_inches='tight')

    # Perform fine-scale simulation.
    ref_Ts = np.zeros((config.Nt_fine, config.N_fine + 2))
    ref_Ts[0] = config.get_T0(config.nodes_fine)
    for i in range(1, config.Nt_fine):
        ref_Ts[i] = physics.simulate(
            config.nodes_fine, config.faces_fine,
            ref_Ts[i - 1], config.T_a, config.T_b,
            config.get_k, config.get_cV, config.rho, config.A,
            config.get_q_hat, np.zeros_like(config.nodes_fine[1:-1]),
            config.dt_fine, config.dt_fine*(i-1), config.dt_fine*i, False
        )

    ref_Ts_downsampled = np.zeros((config.Nt_coarse, config.N_coarse + 2))
    counter = 0
    for time_level in range(0, config.Nt_fine, int(config.dt_coarse / config.dt_fine)):
        idx = npi.indices(np.around(config.nodes_fine, decimals=5),
                          np.around(config.nodes_coarse, decimals=5))
        for i in range(config.N_coarse + 2):
            ref_Ts_downsampled[counter][i] = ref_Ts[time_level][idx[i]]
        counter += 1

    error = unc_Ts - ref_Ts_downsampled
    error_constk = unc_Ts_constk - ref_Ts_downsampled

    plt.figure()
    for index in indices:
        plt.plot(config.nodes_coarse, ref_Ts_downsampled[index], label=index)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(config.results_dir, 'debug_t/ref.pdf'), bbox_inches='tight')

    plt.figure()
    for index in indices:
        plt.plot(config.nodes_coarse, error[index], label=index)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(config.results_dir, 'debug_t/err.pdf'), bbox_inches='tight')

    plt.figure()
    for index in indices:
        plt.plot(config.nodes_coarse, error_constk[index], label=index)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(config.results_dir, 'debug_t/err_const.pdf'), bbox_inches='tight')

########################################################################################################################'

if __name__ == "__main__":
    main()

########################################################################################################################