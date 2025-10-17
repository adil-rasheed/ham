"""
test.py

Written by Sindre Stenen Blakseth, 2021.

Testing if ML-model successfully corrects 1D heat equation simulations.
"""

########################################################################################################################
# Package imports.

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch

########################################################################################################################
# File imports.

from datasets import load_datasets
import models
import physics
import util

########################################################################################################################
# helper function for visualizing test data.

def visualize_test_data(cfg, error_stats_dict, plot_stats_dict):
    # Visualize error stats.
    iterations = np.arange(1, len(error_stats_dict['unc']) + 1, 1)

    plt.figure()
    plt.semilogy(iterations, error_stats_dict['unc'],      'r-', linewidth=2.0, label="Uncorrected")
    plt.semilogy(iterations, error_stats_dict['cor_mean'], 'b-', linewidth=2.0, label="Corrected, mean")
    plt.fill_between(iterations,
                     error_stats_dict['cor_mean'] + error_stats_dict['cor_std'],
                     error_stats_dict['cor_mean'] - error_stats_dict['cor_std'],
                     facecolor='yellow', alpha=0.5, label="Corrected, std.dev.")
    plt.xlim([0, len(error_stats_dict['unc'])])
    plt.xlabel("Test iterations", fontsize=20)
    plt.ylabel(r"$L_2$ Error",    fontsize=20)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.grid()
    plt.legend(prop={'size': 17})
    plt.savefig(os.path.join(cfg.run_dir, "error_stats.pdf"), bbox_inches='tight')
    plt.close()

    if cfg.exact_solution_available:
        t0 = (cfg.train_examples_ratio + cfg.test_examples_ratio)*cfg.t_end
        plot_times = t0 + (cfg.profile_save_steps + 1)*cfg.dt_coarse
        x_dense = np.linspace(cfg.x_a, cfg.x_b, num=1001, endpoint=True)
    if 'time' in plot_stats_dict.keys():
        plot_times = plot_stats_dict['time']

    if 'cor_means_mean' in error_stats_dict.keys():
        labels = ['ML-corrected', 'Uncorrected']
        avgs = [error_stats_dict['cor_means_mean'], error_stats_dict['unc_means_mean']]
        devs = [error_stats_dict['cor_stds_mean'], error_stats_dict['unc_stds_mean']]
        print(labels)
        print("avgs:", avgs)
        print("devs:", devs)

        x = np.arange(len(labels))  # the label locations
        width = 0.4  # the width of the bars

        fig, ax = plt.subplots()
        fig.set_figheight(3)
        fig.set_figwidth(8)
        ax.yaxis.grid(zorder=-1)
        # ax.set_aspect(2)
        rects1 = ax.bar(x - width / 2, avgs, width, label='Average', zorder=3)
        rects2 = ax.bar(x + width / 2, devs, width, label='Standard deviation', zorder=3)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(r'$L_2$ error', fontsize=15)
        # ax.set_title('Scores by group and gender')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.legend(prop={'size': 11})
        ax.set_yscale('log')
        #ax.set_ylim((0, 0.15))
        fig.tight_layout()

        plt.savefig(os.path.join(cfg.run_dir, "histogram.pdf"), bbox_inches='tight')
        plt.close()
        with open(os.path.join(cfg.run_dir, "histogram_data" + ".txt"), "w") as f:
            f.write(str(labels) + "\n")
            f.write("avgs: " + str(avgs) + "\n")
            f.write("devs: " + str(devs) + "\n")

    # Visualize temperature profiles.
    for i in range(plot_stats_dict['unc'].shape[0]):
        plt.figure()
        plt.scatter(plot_stats_dict['x'], plot_stats_dict['unc'][i],
                    s=40, facecolors='none', edgecolors='r', label="Uncorrected")
        plt.scatter(plot_stats_dict['x'], plot_stats_dict['cor_mean'][i], s=40,
                    facecolors='none', edgecolors='b', label="Corrected, mean")
        if cfg.exact_solution_available:
            plt.plot(x_dense, cfg.get_T_exact(x_dense, plot_times[i]), 'k-', linewidth=2.0, label="Reference")
        else:
            plt.plot(plot_stats_dict['x'], plot_stats_dict['ref'][i], 'k-', linewidth=2.0, label="Reference")
        #plt.fill_between(plot_stats_dict['x'],
        #                 plot_stats_dict['cor_mean'][i] + plot_stats_dict['cor_std'][i],
        #                 plot_stats_dict['cor_mean'][i] - plot_stats_dict['cor_std'][i],
        #                 facecolor='yellow', alpha=0.5, label="Corrected, std.dev.")
        plt.xlim([plot_stats_dict['x'][0], plot_stats_dict['x'][-1]])
        plt.xlabel(r"$x$ (m)", fontsize=20)
        plt.ylabel(r"$T$ (K)", fontsize=20)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.grid()
        plt.legend(prop={'size': 17})
        plt.savefig(os.path.join(cfg.run_dir, "profiles" + str(i) + ".pdf"), bbox_inches='tight')
        plt.close()

    # Visualize correction source terms (if applicable).
    if 'src_mean' in plot_stats_dict.keys():
        for i in range(plot_stats_dict['src_mean'].shape[0]):
            plt.figure()
            plt.plot(plot_stats_dict['x'][1:-1], plot_stats_dict['src_mean'][i], 'b-', linewidth=2.0, label="Corrected, mean")
            plt.plot(x_dense,
                     cfg.get_q_hat(x_dense, plot_times[i]) - cfg.get_q_hat_approx(x_dense, plot_times[i]),
                     'k-', linewidth=2.0, label="Reference")
            plt.fill_between(plot_stats_dict['x'][1:-1],
                             plot_stats_dict['src_mean'][i] + plot_stats_dict['src_std'][i],
                             plot_stats_dict['src_mean'][i] - plot_stats_dict['src_std'][i],
                             facecolor='yellow', alpha=0.5, label="Corrected, std.dev.")
            plt.xlim([plot_stats_dict['x'][0], plot_stats_dict['x'][-1]])
            plt.xlabel(r"$x$ (m)", fontsize=20)
            plt.ylabel(r"$T$ (K)", fontsize=20)
            plt.xticks(fontsize=17)
            plt.yticks(fontsize=17)
            plt.grid()
            plt.legend(prop={'size': 17})
            plt.savefig(os.path.join(cfg.run_dir, "src_profiles" + str(i) + ".pdf"), bbox_inches='tight')
            plt.close()

    #plt.show()


########################################################################################################################
# Helper function for saving test data.

def save_test_data(cfg, error_dicts, plot_data_dicts):
    # Pickle raw data.
    with open(os.path.join(cfg.run_dir, "error_data_raw" + ".pkl"), "wb") as f:
        pickle.dump(error_dicts, f)
    with open(os.path.join(cfg.run_dir, "plot_data_raw" + ".pkl"), "wb") as f:
        pickle.dump(plot_data_dicts, f)

    # Save raw error data in a text file for easy manual inspection.
    with open(os.path.join(cfg.run_dir, "error_data_raw" + ".txt"), "w") as f:
        f.write("L2 error (corrected)\t\tL2 error (uncorrected)\n")
        for num, error_dict in enumerate(error_dicts):
            f.write("\nModel instance " + str(num) + "\n")
            for i in range(error_dict['unc'].shape[0]):
                f.write(str(i) + ": " + str(error_dict['cor'][i]) + "\t\t" + str(error_dict['unc'][i]) + "\n")
        f.close()

    # Calculate statistical properties of errors.
    cor_errors = np.asarray([error_dicts[i]['cor'] for i in range(len(error_dicts))])
    cor_error_mean = np.mean(cor_errors, axis=0)
    cor_error_std  = np.std(cor_errors,  axis=0)
    if 'cor_mean' in error_dicts[0].keys():
        cor_means = np.asarray([error_dicts[i]['cor_mean'] for i in range(len(error_dicts))])
        cor_means_mean = np.mean(cor_means, axis=0)
    if 'cor_std' in error_dicts[0].keys():
        cor_stds = np.asarray([error_dicts[i]['cor_std'] for i in range(len(error_dicts))])
        cor_stds_mean = np.mean(cor_stds, axis=0)

    # Calculate statistical properties of plot data
    cor_plots = np.asarray([plot_data_dicts[i]['cor'] for i in range(len(plot_data_dicts))])
    cor_plot_mean = np.mean(cor_plots, axis=0)
    cor_plot_std  = np.std(cor_plots,  axis=0)
    if 'src' in plot_data_dicts[0].keys():
        src_plots = np.asarray([plot_data_dicts[i]['src'] for i in range(len(plot_data_dicts))])
        src_plot_mean = np.mean(src_plots, axis=0)
        src_plot_std  = np.std(src_plots,  axis=0)

    # Pickle statistical properties.
    error_stats_dict = {
        'cor_mean': cor_error_mean,
        'cor_std':  cor_error_std,
        'unc':      error_dicts[0]['unc']
    }
    if 'cor_mean' in error_dicts[0].keys():
        error_stats_dict['cor_means_mean'] = cor_means_mean
    if 'cor_std' in error_dicts[0].keys():
        error_stats_dict['cor_stds_mean'] = cor_stds_mean
    if 'unc_mean' in error_dicts[0].keys():
        error_stats_dict['unc_means_mean'] = error_dicts[0]['unc_mean']
    if 'unc_std' in error_dicts[0].keys():
        error_stats_dict['unc_stds_mean'] = error_dicts[0]['unc_std']
    plot_stats_dict = {
        'cor_mean': cor_plot_mean,
        'cor_std':  cor_plot_std,
        'unc':      plot_data_dicts[0]['unc'], # These are the same for all model instances, ...
        'ref':      plot_data_dicts[0]['ref'], # ... so the choice of retrieving them from ...
        'x':        plot_data_dicts[0]['x'],   # ... the first instance is arbitrary.
        'time':     plot_data_dicts[0]['time']
    }
    if 'src' in plot_data_dicts[0].keys():
        plot_stats_dict['src_mean'] = src_plot_mean
        plot_stats_dict['src_std']  = src_plot_std
    with open(os.path.join(cfg.run_dir, "error_data_stats" + ".pkl"), "wb") as f:
        pickle.dump(error_stats_dict, f)
    with open(os.path.join(cfg.run_dir, "plot_data_stats" + ".pkl"), "wb") as f:
        pickle.dump(plot_stats_dict, f)

    return error_stats_dict, plot_stats_dict


########################################################################################################################
# Testing ML-model.

def single_step_test(cfg, model, num):
    if cfg.model_name[:8] == "Ensemble":
        for m in range(len(model.nets)):
            model.nets[m].net.eval()
    else:
        model.net.eval()

    _, _, dataset_test = load_datasets(cfg, False, False, True)

    unc_tensor = dataset_test[:][0].detach()
    ref_tensor = dataset_test[:][1].detach()
    stats = dataset_test[:6][3].detach().numpy()
    ICs   = dataset_test[:][4].detach().numpy()
    times = dataset_test[:][5].detach().numpy()

    unc_mean = stats[0]
    unc_std  = stats[3]
    ref_mean = stats[1]
    ref_std  = stats[4]
    src_mean = stats[2]
    src_std  = stats[5]

    unc = util.z_unnormalize(unc_tensor.numpy(), unc_mean, unc_std)
    ref = util.z_unnormalize(ref_tensor.numpy(), unc_mean, unc_std)

    L2_errors_unc = np.zeros(cfg.N_test_examples)
    L2_errors_cor = np.zeros(cfg.N_test_examples)

    num_profile_plots = cfg.profile_save_steps.shape[0]
    plot_data_dict = {
        'x': cfg.nodes_coarse,
        'unc': np.zeros((num_profile_plots, cfg.nodes_coarse.shape[0])),
        'ref': np.zeros((num_profile_plots, cfg.nodes_coarse.shape[0])),
        'cor': np.zeros((num_profile_plots, cfg.nodes_coarse.shape[0])),
        'time': np.zeros(num_profile_plots)
    }
    if cfg.model_is_hybrid and cfg.exact_solution_available:
        plot_data_dict['src'] = np.zeros((num_profile_plots, cfg.nodes_coarse.shape[0] - 2))
    for i in range(cfg.N_test_examples):
        old_time = np.around(times[i] - cfg.dt_coarse, decimals=10)
        new_time = np.around(times[i], decimals=10)
        #print("New time:", new_time)

        new_unc = unc[i]
        new_unc_tensor = torch.unsqueeze(unc_tensor[i], 0)
        IC = ICs[i] # The profile at old_time which was used to generate new_unc, which is a profile at new_time.

        new_cor = np.zeros_like(new_unc)
        if cfg.model_name[:8] == "Ensemble":
            if cfg.model_is_hybrid:
                new_src = np.zeros(new_unc.shape[0] - 2)
                for m in range(len(model.nets)):
                    new_src[m] = util.z_unnormalize(torch.squeeze(model.nets[m].net(new_unc_tensor[:,m:m+3]),0).detach().numpy(), src_mean, src_std)
                new_cor = physics.simulate(
                    cfg.nodes_coarse, cfg.faces_coarse,
                    IC, cfg.get_T_a, cfg.get_T_b,
                    cfg.get_k_approx, cfg.get_cV, cfg.rho, cfg.A,
                    cfg.get_q_hat_approx, new_src,
                    cfg.dt_coarse, old_time, new_time, False
                )
            else:
                new_cor[0] = cfg.get_T_a(new_time)  # Since BCs are not ...
                new_cor[-1] = cfg.get_T_b(new_time)  # predicted by the NN.
                for m in range(len(model.nets)):
                    new_cor[m+1] = util.z_unnormalize(torch.squeeze(model.nets[m].net(new_unc_tensor[:,m:m+3]),0).detach().numpy(), src_mean, src_std)
        else:
            if cfg.model_is_hybrid:
                new_src = util.z_unnormalize(torch.squeeze(model.net(new_unc_tensor),0).detach().numpy(), src_mean, src_std)
                new_cor = physics.simulate(
                    cfg.nodes_coarse, cfg.faces_coarse,
                    IC, cfg.get_T_a, cfg.get_T_b,
                    cfg.get_k_approx, cfg.get_cV, cfg.rho, cfg.A,
                    cfg.get_q_hat_approx, new_src,
                    cfg.dt_coarse, old_time, new_time, False
                )
            else:
                new_cor[0] = cfg.get_T_a(new_time)  # Since BCs are not ...
                new_cor[-1] = cfg.get_T_b(new_time)  # predicted by the NN.
                new_cor[1:-1] = util.z_unnormalize(model.net(unc_tensor).detach().numpy(), ref_mean, ref_std)

        #lin_unc = lambda x: util.linearize_between_nodes(x, cfg.nodes_coarse, new_unc)
        #lin_cor = lambda x: util.linearize_between_nodes(x, cfg.nodes_coarse, new_cor)

        if cfg.exact_solution_available:
            #ref_func = lambda x: cfg.get_T_exact(x, new_time)
            new_ref = cfg.get_T_exact(cfg.nodes_coarse, new_time)
        else:
            new_ref = util.z_unnormalize(ref_tensor[i].detach().numpy(), ref_mean, ref_std)
            #ref_func = lambda x: util.linearize_between_nodes(x, cfg.nodes_coarse, new_ref)

        #ref_norm = util.get_L2_norm(cfg.faces_coarse, ref_func)
        #unc_error_norm = util.get_L2_norm(cfg.faces_coarse, lambda x: lin_unc(x) - ref_func(x)) / ref_norm
        #cor_error_norm = util.get_L2_norm(cfg.faces_coarse, lambda x: lin_cor(x) - ref_func(x)) / ref_norm
        ref_norm = util.get_disc_L2_norm(new_ref)
        unc_error_norm = util.get_disc_L2_norm(new_unc - new_ref) / ref_norm
        cor_error_norm = util.get_disc_L2_norm(new_cor - new_ref) / ref_norm

        L2_errors_unc[i] = unc_error_norm
        L2_errors_cor[i] = cor_error_norm

        if i < num_profile_plots:
            plot_data_dict['unc'][i] = new_unc
            plot_data_dict['ref'][i] = new_ref
            plot_data_dict['cor'][i] = new_cor
            if cfg.model_is_hybrid and cfg.exact_solution_available:
                plot_data_dict['src'][i] = new_src
            plot_data_dict['time'][i] = new_time

    error_dict = {
        'unc': L2_errors_unc,
        'cor': L2_errors_cor,
        'unc_mean': np.mean(L2_errors_unc),
        'unc_std': np.std(L2_errors_unc),
        'cor_mean': np.mean(L2_errors_cor),
        'cor_std': np.std(L2_errors_cor)
    }

    return error_dict, plot_data_dict

def simulation_test(cfg, model, num):
    model.net.eval()

    # Get target temperature profile of last validation example. This will be IC for test simulation.
    _, dataset_val, dataset_test = load_datasets(cfg, False, True, True)

    # Get stats used for normalization/unnormalization.
    stats = dataset_test[:6][3].detach().numpy()

    unc_mean = stats[0]
    unc_std  = stats[3]
    ref_mean = stats[1]
    ref_std  = stats[4]
    src_mean = stats[2]
    src_std  = stats[5]

    # Use last reference data example of validation set as IC for test simulation.
    IC = util.z_unnormalize(dataset_val[-1][0].detach().numpy(), ref_mean, ref_std)

    L2_errors_unc = np.zeros(cfg.N_test_examples)
    L2_errors_cor = np.zeros(cfg.N_test_examples)

    plot_steps = cfg.profile_save_steps
    plot_data_dict = {
        'x': cfg.nodes_coarse,
        'unc': np.zeros((plot_steps.shape[0], cfg.nodes_coarse.shape[0])),
        'ref': np.zeros((plot_steps.shape[0], cfg.nodes_coarse.shape[0])),
        'cor': np.zeros((plot_steps.shape[0], cfg.nodes_coarse.shape[0]))
    }
    if cfg.model_is_hybrid and cfg.exact_solution_available:
        plot_data_dict['src'] = np.zeros((plot_steps.shape[0], cfg.nodes_coarse.shape[0] - 2))
    plot_num = 0
    old_unc = IC
    old_cor = IC
    t0 = (cfg.train_examples_ratio + cfg.test_examples_ratio)*cfg.t_end
    diffs = []
    for i in range(cfg.N_test_examples):
        old_time = np.around(t0 + cfg.dt_coarse*i,     decimals=10)
        new_time = np.around(t0 + cfg.dt_coarse*(i+1), decimals=10)

        # new_unc  = new uncorrected profile given old uncorrected profile.
        # new_unc_ = new uncorrected profile given old   corrected profile.
        new_unc = physics.simulate(
            cfg.nodes_coarse, cfg.faces_coarse,
            old_unc, cfg.get_T_a, cfg.get_T_b,
            cfg.get_k_approx, cfg.get_cV, cfg.rho, cfg.A,
            cfg.get_q_hat_approx, np.zeros(cfg.N_coarse),
            cfg.dt_coarse, old_time, new_time, False
        )
        new_unc_ = torch.from_numpy(util.z_normalize(
            physics.simulate(
                cfg.nodes_coarse, cfg.faces_coarse,
                old_cor, cfg.get_T_a, cfg.get_T_b,
                cfg.get_k_approx, cfg.get_cV, cfg.rho, cfg.A,
                cfg.get_q_hat_approx, np.zeros(cfg.N_coarse),
                cfg.dt_coarse, old_time, new_time, False
            ), unc_mean, unc_std
        ))

        new_cor = np.zeros_like(old_cor)
        if cfg.model_is_hybrid:
            new_src = util.z_unnormalize(model.net(new_unc_).detach().numpy(), src_mean, src_std)
            new_cor = physics.simulate(
                cfg.nodes_coarse, cfg.faces_coarse,
                old_cor, cfg.get_T_a, cfg.get_T_b,
                cfg.get_k_approx, cfg.get_cV, cfg.rho, cfg.A,
                cfg.get_q_hat_approx, new_src,
                cfg.dt_coarse, old_time, new_time, False
            )
        else:
            new_cor[0]  = cfg.get_T_a(new_time)   # Since BCs are not ...
            new_cor[-1] = cfg.get_T_b(new_time)  # predicted by the NN.
            new_cor[1:-1] = util.z_unnormalize(model.net(new_unc_).detach().numpy(), ref_mean, ref_std)

        #lin_unc = lambda x: util.linearize_between_nodes(x, cfg.nodes_coarse, new_unc)
        #lin_cor = lambda x: util.linearize_between_nodes(x, cfg.nodes_coarse, new_cor)


        if cfg.exact_solution_available:
            #ref_func = lambda x: cfg.get_T_exact(x, new_time)
            new_ref = cfg.get_T_exact(cfg.nodes_coarse, new_time)
        else:
            new_ref_tensor = dataset_test[i][1]
            new_ref = util.z_unnormalize(new_ref_tensor.detach().numpy(), ref_mean, ref_std)
            #ref_func = lambda x: util.linearize_between_nodes(x, cfg.nodes_coarse, new_ref)

        #ref_norm = util.get_L2_norm(cfg.faces_coarse, ref_func)
        #unc_error_norm = util.get_L2_norm(cfg.faces_coarse, lambda x: lin_unc(x) - ref_func(x)) / ref_norm
        #cor_error_norm = util.get_L2_norm(cfg.faces_coarse, lambda x: lin_cor(x) - ref_func(x)) / ref_norm
        ref_norm = util.get_disc_L2_norm(new_ref)
        unc_error_norm = util.get_disc_L2_norm(new_unc - new_ref) / ref_norm
        cor_error_norm = util.get_disc_L2_norm(new_cor - new_ref) / ref_norm

        L2_errors_unc[i] = unc_error_norm
        L2_errors_cor[i] = cor_error_norm

        if i in plot_steps:
            plot_data_dict['unc'][plot_num] = new_unc
            plot_data_dict['ref'][plot_num] = new_ref
            plot_data_dict['cor'][plot_num] = new_cor
            if cfg.model_is_hybrid and cfg.exact_solution_available:
                plot_data_dict['src'][plot_num] = new_src
            plot_data_dict['time'][plot_num] = new_time
            plot_num += 1

        if i % 10 == 0 and i <= 50:
            diffs.append(np.asarray(new_cor - new_ref))

        old_unc = new_unc
        old_cor = new_cor

    error_dict = {
        'unc': L2_errors_unc,
        'cor': L2_errors_cor
    }

    plt.figure()
    for i in range(len(diffs)):
        plt.plot(cfg.nodes_coarse, diffs[i], label=str(10*i))
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(cfg.run_dir, "differences.pdf"), bbox_inches='tight')
    plt.close()

    return error_dict, plot_data_dict

########################################################################################################################

def main():
    model = models.create_new_model(None, None)
    num = 0
    simulation_test(None, model, num)

########################################################################################################################

if __name__ == "__main__":
    main()

########################################################################################################################