"""
parameter_grid_search.py

Written by Sindre Stenen Blakseth, 2021.

Script for performing parameter grid search.
"""

########################################################################################################################
# Package imports.

import itertools
import numpy as np
import os
import torch

########################################################################################################################
# File imports.

from datasets import load_datasets_from_path
import models
import train

########################################################################################################################
# Parameter space.

def grid_search(cfg):
    os.makedirs(cfg.run_dir, exist_ok=False)
    # Load datasets and create dataloaders.
    dataset_paths = [
        [os.path.join(cfg.datasets_dir, 's2B_no-aug_sst_train.pt'),
         os.path.join(cfg.datasets_dir, 's2B_no-aug_sst_val.pt'),
         os.path.join(cfg.datasets_dir, 's2B_no-aug_sst_test.pt')],
        [os.path.join(cfg.datasets_dir, 's2B_no-aug_sst_train.pt'),
         os.path.join(cfg.datasets_dir, 's2B_no-aug_sst_val.pt'),
         os.path.join(cfg.datasets_dir, 's2B_no-aug_sst_test.pt')],
        [os.path.join(cfg.datasets_dir, 's2B_no-aug_sst_train.pt'),
         os.path.join(cfg.datasets_dir, 's2B_no-aug_sst_val.pt'),
         os.path.join(cfg.datasets_dir, 's2B_no-aug_sst_test.pt')]
    ]
    dataloaders = []
    for i in range(len(dataset_paths)):
        train_set, val_set, test_set = load_datasets_from_path(
            dataset_paths[i][0], dataset_paths[i][1], dataset_paths[i][2]
        )
        dataloader_train = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=cfg.batch_size_train,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        dataloader_val = torch.utils.data.DataLoader(
            dataset=val_set,
            batch_size=cfg.batch_size_val,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        dataloader_test = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=cfg.batch_size_test,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        dataloaders.append([dataloader_train, dataloader_val, dataloader_test])

    search_data = []

    axes   = None
    labels = None
    if cfg.model_name == "GlobalDense":
        learning_rates = [1e-4, 1e-5]
        dropout_probs = [0.0, 0.2]
        widths = (cfg.N_coarse * np.asarray([2, 3, 4, 5, 7])).astype(int)
        depths = [3, 4, 5, 6, 8, 10]
        axes   = [learning_rates, dropout_probs, widths, depths]
        labels = "learning rate,\tdropout prob.,\twidth,\tdepth"
    elif cfg.model_name == "GlobalCNN":
        learning_rates = [1e-4, 1e-5]
        conv_nums = [3, 5, 7, 9, 12]
        channel_nums = [10, 15, 20, 25, 30, 40]
        fc_nums = [1, 2]
        axes = [learning_rates, conv_nums, channel_nums, fc_nums]
        labels = "learning rate,\tNo. conv layers,  No. conv channels,  No. fc layers"
    elif cfg.model_name == "LocalDense":
        learning_rates = [1e-4, 1e-5]
        dropout_probs = [0.0, 0.2]
        widths = [3, 5, 7, 10, 15]
        depths = [3, 5, 7, 10, 15]
        axes = [learning_rates, dropout_probs, widths, depths]
        labels = "learning rate,\tdropout prob.,\twidth,\tdepth"
    elif cfg.model_name == "EnsembleLocalDense":
        learning_rates = [1e-4, 1e-5]
        dropout_probs = [0.0, 0.2]
        widths = [3, 5, 7, 10, 15]
        depths = [3, 5, 7, 10, 15]
        axes = [learning_rates, dropout_probs, widths, depths]
        labels = "learning rate,\tdropout prob.,\twidth,\tdepth"
    elif cfg.model_name == "EnsembleGlobalCNN":
        learning_rates = [1e-4, 1e-5]
        conv_nums = [3, 5, 7, 9, 12]
        channel_nums = [10, 15, 20, 25, 30, 40]
        axes = [learning_rates, conv_nums, channel_nums]
        labels = "learning rate,\tNo. conv layers,  No. conv channels"

    combos = list(itertools.product(*axes))

    for combo in combos:
        print("\n-------------------------------------------------------")
        print(labels)
        print(combo)
        print("")

        if cfg.model_name == "GlobalDense":
            cfg.learning_rate = combo[0]
            cfg.dropout_prob = combo[1]
            cfg.hidden_layer_size = combo[2]
            cfg.num_layers = combo[3]
        elif cfg.model_name == "GlobalCNN":
            cfg.learning_rate = combo[0]
            cfg.num_conv_layers = combo[1]
            cfg.num_channels = combo[2]
            cfg.num_fc_layers = combo[3]
        elif cfg.model_name == "LocalDense":
            cfg.learning_rate = combo[0]
            cfg.dropout_prob = combo[1]
            cfg.hidden_layer_size = combo[2]
            cfg.num_layers = combo[3]
        elif cfg.model_name == "EnsembleLocalDense":
            cfg.learning_rate = combo[0]
            cfg.dropout_prob = combo[1]
            cfg.hidden_layer_size = combo[2]
            cfg.num_layers = combo[3]
        elif cfg.model_name == "EnsembleGlobalCNN":
            cfg.learning_rate = combo[0]
            cfg.num_conv_layers = combo[1]
            cfg.num_channels = combo[2]

        final_val_losses = np.zeros(len(dataloaders))

        for system_num in range(len(dataloaders)):
            print("System num:", system_num)
            model = models.create_new_model(cfg, cfg.get_model_specific_params())
            data_dict = train.train(cfg, model, 0, dataloaders[system_num][0], dataloaders[system_num][1])
            final_val_losses[system_num] = data_dict["Validation loss"][1][-1]

        final_val_losses_sum = np.sum(final_val_losses)
        search_data.append({"str": labels + "\n" + str(combo), "sum": final_val_losses_sum, "losses": final_val_losses})

    lowest_loss = np.inf
    best_params = ""

    with open(os.path.join(cfg.run_dir, "grid_search_results" + ".txt"), "w") as f:
        print("\n")
        print("Results from parameter grid search for model " + cfg.model_name + "\n")
        f.write("Results from parameter grid search for model " + cfg.model_name + "\n\n")
        for data_dict in search_data:
            print("Parameters:\t", data_dict["str"])
            print("Losses:\t", data_dict["losses"])
            print("Loss sum:\t", data_dict["sum"])
            print("\n")
            f.write("Parameters:\t" + str(data_dict["str"]) + "\n")
            f.write("Losses:\t" + str(data_dict["losses"]) + "\n")
            f.write("Loss sum:\t" + str(data_dict["sum"]) + "\n")
            f.write("\n")
            if data_dict["sum"] < lowest_loss:
                lowest_loss = data_dict["sum"]
                best_params = data_dict["str"]

        print("BEST PARAMETERS:")
        print(best_params)

        f.write("BEST PARAMETERS:\n")
        f.write(best_params)


########################################################################################################################

def main():
    pass

########################################################################################################################

if __name__ == "__main__":
    main()

########################################################################################################################