"""
determine_fine_resolution.py

Written by Sindre Stenen Blakseth, 2021.

Script for determining the appropriate fine resolution for System 1 and System 2
"""

########################################################################################################################
# Package imports.

import numpy as np

########################################################################################################################
# File imports.

import config
import physics
import util

########################################################################################################################
# Verify results from main.

def verify(N, dt):
    # Solution predicted by main.
    N_ref = N
    dx_ref = (config.x_b - config.x_a) / N_ref
    faces_ref = np.linspace(config.x_a, config.x_b, num=N_ref + 1, endpoint=True)
    nodes_ref = np.zeros(N_ref + 2)
    nodes_ref[0] = config.x_a
    nodes_ref[1:-1] = faces_ref[:-1] + dx_ref / 2
    nodes_ref[-1] = config.x_b
    dt_ref = dt

    T_ref = physics.simulate(nodes_ref, faces_ref,
                             config.get_T0(nodes_ref), config.get_T_a, config.get_T_b,
                             config.get_k, config.get_cV, config.rho, config.A,
                             config.get_q_hat, np.zeros_like(nodes_ref[1:-1]),
                             dt_ref, 0, config.t_end, steady=False)

    # Improved spatial resolution.
    N_space = N_ref * 2
    dx_space = (config.x_b - config.x_a) / N_space
    faces_space = np.linspace(config.x_a, config.x_b, num=N_space + 1, endpoint=True)
    nodes_space = np.zeros(N_space + 2)
    nodes_space[0] = config.x_a
    nodes_space[1:-1] = faces_space[:-1] + dx_space / 2
    nodes_space[-1] = config.x_b

    T_space = physics.simulate(nodes_space, faces_space,
                               config.get_T0(nodes_space), config.get_T_a, config.get_T_b,
                               config.get_k, config.get_cV, config.rho, config.A,
                               config.get_q_hat, np.zeros_like(nodes_ref[1:-1]),
                               dt_ref, 0, config.t_end, steady=False)

    # Improved temporal resolution.
    dt_time = dt / 2.0

    T_time = physics.simulate(nodes_ref, faces_ref,
                              config.get_T0(nodes_ref), config.get_T_a, config.get_T_b,
                              config.get_k, config.get_cV, config.rho, config.A,
                              config.get_q_hat, np.zeros_like(nodes_ref[1:-1]),
                              dt_time, 0, config.t_end, steady=False)

    lin_T_ref   = lambda x: util.linearize_between_nodes(x, nodes_ref, T_ref)
    lin_T_space = lambda x: util.linearize_between_nodes(x, nodes_space, T_space)
    lin_T_time  = lambda x: util.linearize_between_nodes(x, nodes_ref, T_time)

    diff_space = lambda x: lin_T_ref(x) - lin_T_space(x)
    diff_time  = lambda x: lin_T_ref(x) - lin_T_time(x)

    norm_space = util.get_L2_norm(faces_ref, diff_space) / util.get_L2_norm(faces_ref, lin_T_ref)
    norm_time  = util.get_L2_norm(faces_ref, diff_time)  / util.get_L2_norm(faces_ref, lin_T_ref)

    print("Spatial  resolution difference:", norm_space)
    print("Temporal resolution difference:", norm_time )

########################################################################################################################

def main():
    # Stop grid refinement when relative difference between current and previous solution is less than tol.
    tol = 1e-7

    max_sequential_temporal_updates = np.inf

    # Initialize
    N_old     = config.N_coarse
    dx_old    = config.dx_coarse
    faces_old = config.faces_coarse
    nodes_old = config.nodes_coarse
    dt_old    = config.dt_coarse

    T_old     = physics.simulate(nodes_old, faces_old,
                                  config.get_T0(nodes_old), config.get_T_a, config.get_T_b,
                                  config.get_k, config.get_cV, config.rho, config.A,
                                 config.get_q_hat, np.zeros_like(nodes_old[1:-1]),
                                  dt_old, 0, config.t_end, steady=False)

    cont_refine  = True
    refine_space = True
    refine_time  = True
    while cont_refine:
        space_was_refined = False
        while refine_space:
            N_new     = N_old * 2
            dx_new    = (config.x_b - config.x_a) / N_new
            faces_new = np.linspace(config.x_a, config.x_b, num=N_new + 1, endpoint=True)
            nodes_new = np.zeros(N_new + 2)
            nodes_new[0]    = config.x_a
            nodes_new[1:-1] = faces_new[:-1] + dx_new / 2
            nodes_new[-1]   = config.x_b
            dt_new = dt_old

            print("N_new:", N_new)

            T_new = physics.simulate(nodes_new, faces_new,
                                     config.get_T0(nodes_new), config.get_T_a, config.get_T_b,
                                     config.get_k, config.get_cV, config.rho, config.A,
                                     config.get_q_hat, np.zeros_like(nodes_new[1:-1]),
                                     dt_new, 0, config.t_end, steady=False)

            lin_T_new = lambda x: util.linearize_between_nodes(x, nodes_new, T_new)
            lin_T_old = lambda x: util.linearize_between_nodes(x, nodes_old, T_old)

            norm = util.get_L2_norm(config.faces_coarse, lambda x: lin_T_new(x) - lin_T_old(x)) / util.get_L2_norm(config.faces_coarse, lin_T_old)
            print("norm:", norm)

            if norm < tol:
                refine_space = False
                refine_time  = True
            else:
                N_old     = N_new
                dx_old    = dx_new
                faces_old = faces_new
                nodes_old = nodes_new
                dt_old    = dt_new
                T_old     = T_new
                space_was_refined = True

        if not space_was_refined:
            cont_refine = False
            continue
        time_was_refined = False
        sequential_temporal_udpates = 0
        while refine_time and sequential_temporal_udpates < max_sequential_temporal_updates:
            N_new = N_old
            dx_new = (config.x_b - config.x_a) / N_new
            faces_new = np.linspace(config.x_a, config.x_b, num=N_new + 1, endpoint=True)
            nodes_new = np.zeros(N_new + 2)
            nodes_new[0] = config.x_a
            nodes_new[1:-1] = faces_new[:-1] + dx_new / 2
            nodes_new[-1] = config.x_b
            dt_new = dt_old / 2.0

            T_new = physics.simulate(nodes_new, faces_new,
                                     config.get_T0(nodes_new), config.get_T_a, config.get_T_b,
                                     config.get_k, config.get_cV, config.rho, config.A,
                                     config.get_q_hat, np.zeros_like(nodes_new[1:-1]),
                                     dt_new, 0, config.t_end, steady=False)

            lin_T_new = lambda x: util.linearize_between_nodes(x, nodes_new, T_new)
            lin_T_old = lambda x: util.linearize_between_nodes(x, nodes_old, T_old)

            norm = util.get_L2_norm(config.faces_coarse, lambda x: lin_T_new(x) - lin_T_old(x)) / util.get_L2_norm(
                config.faces_coarse, lin_T_old)
            print("norm:", norm)

            print("dt_new:", dt_new)

            if norm < tol:
                refine_space = True
                refine_time = False
            else:
                N_old     = N_new
                dx_old    = dx_new
                faces_old = faces_new
                nodes_old = nodes_new
                dt_old = dt_new
                T_old  = T_new
                time_was_refined = True
                sequential_temporal_udpates += 1

        if not time_was_refined:
            cont_refine = False
        else:
            refine_space = True

    print("Coarsest satisfactory grid discretization is defined by")
    print("N_fine:", N_old)
    print("dt:", dt_old)

    #verify(int(N_old), dt_old)

    #plt.figure()
    #plt.plot(nodes_old, T_old)
    #plt.show()

########################################################################################################################

if __name__ == "__main__":
    main()