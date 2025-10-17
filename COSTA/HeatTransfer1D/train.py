"""
train.py

Written by Sindre Stenen Blakseth, 2021.

Script for training models for data-driven corrections of the 1D heat equation.
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

########################################################################################################################
# Training ML-model.

def train(cfg, model, num, dataloader_train, dataloader_val):
    it_per_epoch = len(dataloader_train)
    num_epochs = cfg.max_train_it // it_per_epoch + 1

    it = 0

    lowest_val_los = np.inf
    val_epoch_since_improvement = 0

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader_train):
            if it >= cfg.max_train_it or (val_epoch_since_improvement >= cfg.overfit_limit and it >= cfg.min_train_it):
                break

            it += 1

            unc_data = data[0] # unc = uncorrected.
            ref_data = data[1] # ref = reference.
            src_data = data[2] # src = source.

            if cfg.model_name[:8] == "Ensemble":
                for m in range(len(model.nets)):
                    model.nets[m].net.train()

                    unc_stencil = unc_data[:, m:m + 3].clone()
                    ref_stencil = ref_data[:, m + 1:m + 2].clone()
                    src_stencil = src_data[:, m:m + 1].clone()
                    out_stencil = model.nets[m].net(unc_stencil)

                    if cfg.model_is_hybrid:
                        loss = model.nets[m].loss(out_stencil, src_stencil)
                    else:
                        loss = model.nets[m].loss(out_stencil, ref_stencil)

                    if it % cfg.print_train_loss_period == 0 and m == 0:
                        print(it, loss.item())
                    if it % cfg.save_train_loss_period == 0:
                        model.nets[m].train_losses.append(loss.item())
                        model.nets[m].train_iterations.append(it)

                    model.nets[m].net.zero_grad()

                    loss.backward()

                    model.nets[m].optimizer.step()

                if it % cfg.validation_period == 0:
                    total_val_loss = 0.0
                    for m in range(len(model.nets)):
                        model.nets[m].net.eval()
                        with torch.no_grad():
                            for j, val_data in enumerate(dataloader_val):
                                unc_data_val = val_data[0][:,m:m+3].clone()
                                ref_data_val = val_data[1][:,m+1:m+2].clone()
                                src_data_val = val_data[2][:,m:m+1].clone()
                                out_data_val = model.nets[m].net(unc_data_val)
                                if cfg.model_is_hybrid:
                                    val_loss = model.nets[m].loss(out_data_val, src_data_val)
                                else:
                                    val_loss = model.nets[m].loss(out_data_val, ref_data_val)
                                total_val_loss += val_loss
                                model.nets[m].val_losses.append(val_loss.item())
                                model.nets[m].val_iterations.append(it)
                    #print(it, "val", total_val_loss)
                    if total_val_loss < lowest_val_los:
                        lowest_val_los = total_val_loss
                        val_epoch_since_improvement = 0
                    else:
                        val_epoch_since_improvement += 1
            else:
                model.net.train()

                out_data = model.net(unc_data) # out = output (corrected profile or predicted correction source term).

                if cfg.model_is_hybrid:
                    loss = model.loss(out_data, src_data)
                else:
                    loss = model.loss(out_data, ref_data[:, 1:-1])

                """
                if it == cfg.num_train_it:
                    print("unc_data:", unc_data)
                    print("ref_data:", ref_data)
                    print("src_data:", src_data)
                    print("out_data:", out_data)
                    print("loss:", loss)
                """

                if it % cfg.print_train_loss_period == 0:
                    print(it, loss.item())
                if it % cfg.save_train_loss_period == 0:
                    model.train_losses.append(loss.item())
                    model.train_iterations.append(it)

                model.net.zero_grad()

                loss.backward()

                model.optimizer.step()

                if it % cfg.validation_period == 0:
                    model.net.eval()
                    with torch.no_grad():
                        for j, val_data in enumerate(dataloader_val):
                            unc_data_val = val_data[0]
                            ref_data_val = val_data[1]
                            src_data_val = val_data[2]
                            out_data_val = model.net(unc_data_val)
                            if cfg.model_is_hybrid:
                                val_loss = model.loss(out_data_val, src_data_val)
                            else:
                                val_loss = model.loss(out_data_val, ref_data_val[:, 1:-1])
                            model.val_losses.append(val_loss.item())
                            model.val_iterations.append(it)
                            if val_loss < lowest_val_los:
                                lowest_val_los = val_loss
                                val_epoch_since_improvement = 0
                            else:
                                val_epoch_since_improvement += 1

    if cfg.model_name[:8] == "Ensemble":
        train_losses = np.zeros((len(model.nets), len(model.nets[0].train_losses)))
        for m in range(len(model.nets)):
            train_losses[m] = model.nets[m].train_losses
        train_losses = np.sum(train_losses, axis=0)
        #print("train_losses.shape:", train_losses.shape)
        val_losses = np.zeros((len(model.nets), len(model.nets[0].val_losses)))
        for m in range(len(model.nets)):
            #print("model.nets[m].val_losses:", model.nets[m].val_losses)
            val_losses[m] = model.nets[m].val_losses
        val_losses = np.sum(val_losses, axis=0)
        #print("val_losses.shape:", val_losses.shape)
        train_iterations = model.nets[0].train_iterations
        val_iterations = model.nets[0].val_iterations
        #print("len(val_iterations):", len(val_iterations))
    else:
        train_losses = model.train_losses
        val_losses = model.val_losses
        train_iterations = model.train_iterations
        val_iterations = model.val_iterations

    fig1, ax1 = plt.subplots()
    ax1.set_title("Training and Validation Loss, Model " + str(num))
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("MSELoss")
    ax1.plot(train_iterations, train_losses, label='Training loss')
    ax1.plot(val_iterations, val_losses, label='Validation loss')
    ax1.set_yscale('log')
    ax1.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(cfg.run_dir, "training_and_val_loss" + str(num) + ".pdf"))
    plt.close()
    #print("Ticks:", np.arange(np.log10(min_loss), np.log(max_loss) + 1, 1.0))

    data_dict = dict()
    data_dict['Training loss'] = np.asarray([train_iterations, train_losses])
    data_dict['Validation loss'] = np.asarray([val_iterations, val_losses])
    pickle.dump(data_dict, open(os.path.join(cfg.run_dir, "plot_data_loss_training_and_val" + str(num) + ".pkl"), "wb"))

    return data_dict

########################################################################################################################