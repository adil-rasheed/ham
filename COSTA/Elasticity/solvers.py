import numpy as np
#import pandas as pd
import tensorflow as tf
import datetime
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt, cm
plt.rc('font', size=13) #increase font text size from 10
import json


# for tracking memory usage
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

import tracemalloc
tracemalloc.start()

from pympler import asizeof
def memory_check(object):
    total_val = 0
    print(f'\nExposing object')
    for key, val in sorted(((name, asizeof.asizeof(val)) for name, val in object.__dict__.items()), key=lambda x: -x[1]):
        if val > 200:
            print(sizeof_fmt(val), key)
        total_val += val
    print('total val:', sizeof_fmt(total_val))

def memory_check_global():
    total_val = 0
    globals_stored = set(globals())
    print("\nGlobal Variables:")
    for key, val in sorted(((key, asizeof.asizeof(eval(key))) for key in globals_stored), key=lambda x: -x[1]):
        if val > 200:
            print(sizeof_fmt(val), key, type(eval(key)))
        total_val += val
    print('total val:', sizeof_fmt(total_val))


import FEM
from utils import merge_first_dims
import methods

COLORS = { # Note some have same colors, should not be used at the same time
        'exact':'k', # blacK
        'FEM':(1,0,0),# red
        'DNN':(0.9,0.6,0), #gold
        'pgDNN':(1,1,0), # yellow
        'CoSTA_DNN': (0,0,1), # blue
        'CoSTA_pgDNN':(0,0.5,0), # Green
        'LSTM':(.86,0,1), # purple-pink
        'pgLSTM':(0.6,0.2,0),
        'pgLR':(0.6,0.2,0), # WARNING: SAME AS ABOVE
        'CoSTA_LSTM':'c',
        'CoSTA_pgLSTM':(0.67,1,0),
        'CoSTA_pgLR':(0.67,1,0), # WARNING: SAME AS ABOVE
        'FEM_2':(0,0.5,0), # Green # WARNING: SAME AS CoSTA_pgDNN
        }



#############################################################################
###   Solver class compares the different solvers defined in methods.py   ###
#############################################################################
class Solvers:
    def __init__(self, sol, equation='heat', models=None, modelnames=None, Ne=10, time_steps=20, p=1, xa=0, xb=1, ya=0, yb=1, dim=1, static=False, skip_create_data=False, NNkwargs={}):
        '''
        either models or modelnames must be specified, not both
        models - list of models
        modelnames - dict of names of models to be created, and how many of each
        '''
        assert models == None or modelnames == None
        assert models != None or modelnames != None

        self.sol = sol # solution
        self.disc = FEM.Disc(
                equation=equation,
                T=sol.T, 
                time_steps=time_steps, 
                Ne=Ne,
                p=p,
                dim=dim,
                xa=xa,
                xb=xb,
                ya=ya,
                yb=yb,
                static=False,
                )
        self.alpha_train = [.1,.2,.3,.4,.5,.6,.9,1,1.2,1.3,1.4,1.6,1.7,1.8,1.9,2]
        self.alpha_val = [.8,1.1]
        self.alpha_test_interpol = [.7,1.5]
        self.alpha_test_extrapol = [-0.5,2.5]
        self.plot = True

        if modelnames != None:
            self.modelnames = modelnames
            self.models = []
            for modelname in modelnames:
                for i in range(modelnames[modelname]):
                    if modelname == 'DNN':
                        self.models.append(methods.DNN_solver(disc=self.disc, **(NNkwargs[modelname])))
                    elif modelname == 'pgLR':
                        self.models.append(methods.pgLR_solver(disc=self.disc, **(NNkwargs[modelname])))
                    elif modelname == 'pgDNN':
                        self.models.append(methods.pgDNN_solver(disc=self.disc, **(NNkwargs[modelname])))
                    elif modelname == 'LSTM':
                        self.models.append(methods.LSTM_solver(disc=self.disc, **(NNkwargs[modelname])))
                    elif modelname == 'pgLSTM':
                        self.models.append(methods.pgLSTM_solver(disc=self.disc, **(NNkwargs[modelname])))
                    elif modelname == 'CoSTA_DNN':
                        self.models.append(methods.CoSTA_DNN_solver(disc=self.disc,**(NNkwargs[modelname])))
                    elif modelname == 'CoSTA_pgLR':
                        self.models.append(methods.CoSTA_pgLR_solver(disc=self.disc,**(NNkwargs[modelname])))
                    elif modelname == 'CoSTA_pgDNN':
                        self.models.append(methods.CoSTA_pgDNN_solver(disc=self.disc,**(NNkwargs[modelname])))
                    elif modelname == 'CoSTA_LSTM':
                        self.models.append(methods.CoSTA_LSTM_solver(disc=self.disc, **(NNkwargs[modelname])))
                    elif modelname == 'CoSTA_pgLSTM':
                        self.models.append(methods.CoSTA_pgLSTM_solver(disc=self.disc, **(NNkwargs[modelname])))
                    else:
                        print(f'WARNING!!!! model named {modelname} not implemented')
                #print(modelname)
                #self.models[-1].model.summary()

        if models != None:
            self.models = models
            self.modelnames = {}
            for model in models:
                if model.name in self.modelnames:
                    self.modelnames[model.name] += 1
                else:
                    self.modelnames[model.name] = 1
        if not skip_create_data:
            self.create_data()



    def __data_set(self,alphas, silence=False):
        ham_data=np.zeros((self.disc.time_steps,len(alphas),self.disc.Nv + len(self.disc.inner_ids2))) # data for ham NN model
        pnn_data=np.zeros((self.disc.time_steps,len(alphas),self.disc.Nv + len(self.disc.inner_ids2))) # data for pure NN model
        extra_feats=np.zeros((self.disc.time_steps,len(alphas),2)) # alpha and time
        for i, alpha in enumerate(alphas):
            if not silence:
                print(f'--- making data set for alpha {alpha} ---')
            self.sol.set_alpha(alpha)
            fem_model = self.disc.make_model(self.sol.f, self.sol.u, w_ex=self.sol.w)
            for t in range(self.disc.time_steps):
                ex_step = self.disc.format_u(fem_model.u_ex(fem_model.pts, t=fem_model.time)) # exact solution before step
                pnn_data[t,i,:self.disc.Nv] = ex_step
                fem_model.u_fem = ex_step # Use u_ex as u_prev
                if self.disc.equation == 'elasticity':
                    if fem_model.time == 0:
                        fem_model.w_fem = self.disc.format_u(fem_model.w_ex(fem_model.pts, t=fem_model.time))
                    else:
                        fem_model.w_fem = self.disc.format_u(fem_model.u_ex(fem_model.pts, t=fem_model.time)-fem_model.u_ex(fem_model.pts, t=fem_model.time-fem_model.k))/fem_model.k
                fem_model.step() # make a step
                ham_data[t,i,:self.disc.Nv] = fem_model.u_fem
                extra_feats[t,i,0] = alpha
                extra_feats[t,i,1] = fem_model.time
                ex_step = self.disc.format_u(fem_model.u_ex(fem_model.pts, t=fem_model.time)) # exact solution after step
                error = ex_step - fem_model.u_fem
                pnn_data[t,i, self.disc.Nv:] = ex_step[self.disc.inner_ids2]
                if self.disc.equation == 'elasticity':
                    ham_data[t,i, self.disc.Nv:] = (fem_model.MA @ np.concatenate([error,error/fem_model.k]))[self.disc.inner_ids2] # residual
                else:
                    ham_data[t,i, self.disc.Nv:] = (fem_model.MA @ error)[self.disc.inner_ids2] # residual
        return {'ham':ham_data, 'pnn':pnn_data, 'extra_feats':extra_feats}


    def create_data(self, silence = False):
        start_time = datetime.datetime.now()
        self.train_data = self.__data_set(self.alpha_train, silence)
        self.val_data = self.__data_set(self.alpha_val, silence)

        plot_one_step_sources = False
        if plot_one_step_sources:
            self.test_data = self.__data_set(self.alpha_val) # for plotting learnt source term only !
        print(f'\nTime making data set: {datetime.datetime.now()-start_time}')

    def load_weights(self, model_folder):
        for j, model in enumerate(self.models):
            model.model.load_weights(model_folder+f'{j}_{model.name}_{self.disc.time_steps}_{self.sol.name}')
            model.train(self.train_data, self.val_data, True) # For setting mean/var, it wont fit the model
            # Show memory usage
            #memory_check(self)
            #memory_check(model)
            #memory_check(model.model)
        print('model weights loaded')

    def train(self, model_folder=None, figname=None):
        start_time = datetime.datetime.now()

        # prepare plotting
        prev_name = ''
        i=-1


        for j, model in enumerate(self.models):

            # train
            model_start_time = datetime.datetime.now()
            model.train(self.train_data, self.val_data)
            if model_folder != None:
                model.model.save_weights(model_folder+f'{j}_{model.name}_{self.disc.time_steps}_{self.sol.name}')
            print(f'\nTime training model "{model.name}": {datetime.datetime.now()-model_start_time}')

            # plot history
            if model.name != prev_name:
                prev_name = model.name
                i+=1
                if i>len(self.modelnames):
                    print("Models is not sorted by name") # note testing also needs this
                    1/0
                epochs_vector = np.arange(1, len(model.train_hist)+1)
                plt.plot(epochs_vector, model.train_hist, '--', color=COLORS[model.name], label=f'{model.name},train')
                plt.plot(epochs_vector, model.val_hist, color=COLORS[model.name], label=f'{model.name},val')
            else:
                epochs_vector = np.arange(1, len(model.train_hist)+1)
                plt.plot(epochs_vector, model.train_hist, '--', color=COLORS[model.name])
                plt.plot(epochs_vector, model.val_hist, color=COLORS[model.name])

            if model_folder != None:
                # save history json file
                with open(model_folder+f'{j}_{model.name}_{self.disc.time_steps}_{self.sol.name}.json', "w") as f:
                    f.write(json.dumps({
                        'train':model.train_hist, 
                        'val':model.val_hist
                        }))

                # workaround: reload weights, to forget unnecessary stuff filling up memomry
                model.resetmodel()
                model.model.load_weights(model_folder+f'{j}_{model.name}_{self.disc.time_steps}_{self.sol.name}')
            model.train(self.train_data, self.val_data, True) # For setting mean/var, it wont fit the model
            plt.yscale('log')
            plt.xlabel('Number of epochs')
            plt.ylabel('Loss (MSE)')
            plt.legend(title='losses')
            plt.grid()
            
            ## Show memory usage
            ##memory_check_global()
            #memory_check(self)
            ##memory_check(model)
            ##memory_check(model.model)

            #snapshot = tracemalloc.take_snapshot()
            #top_stats = snapshot.statistics('lineno')
            #fld = '/home/sir/Desktop/FEM_CoSTA/'
            #for stat in top_stats:
            #    if stat[:len(fld)] == fld:
            #        print(stat[len(fld):])

        print(f'\nTime training all models: {datetime.datetime.now()-start_time}')

        plt.tight_layout()
        if figname != None:
            plt.savefig(figname+'.pdf')
        if self.plot:
            plt.show()
        else:
            plt.close()

        #if plot_one_step_sources:
        #    # Plot some source terms
        #    for t in range(3):
        #        plt.plot(self.disc.pts[self.disc.inner_ids1], Y[t], 'k',  label='exact source')
        #        for model in self.hamNNs:
        #            plt.plot(self.disc.pts[self.disc.inner_ids1], model(np.array([X[t]]))[0], 'r--',  label='ham source')
        #        plt.legend(title='trainset')
        #        plt.grid()
        #        plt.savefig(f'../preproject/1d_heat_figures/{self.sol}_source_train_{t}.pdf')
        #        plt.show()

        #        plt.plot(self.disc.pts[self.disc.inner_ids1], Y_val[t], 'k',  label='exact source')
        #        for model in self.hamNNs:
        #            plt.plot(self.disc.pts[self.disc.inner_ids1], model(np.array([X_val[t]]))[0], 'r--',  label='ham source')
        #        plt.legend(title='valset')
        #        plt.grid()
        #        plt.savefig(f'../preproject/1d_heat_figures/{self.sol}_source_val_{t}.pdf')
        #        plt.show()

        #        plt.plot(self.disc.pts[self.disc.inner_ids1], Y_test[t], 'k',  label='exact source')
        #        for model in self.hamNNs:
        #            plt.plot(self.disc.pts[self.disc.inner_ids1], model(np.array([X_test[t]]))[0], 'r--',  label='ham source')
        #        plt.legend(title='testset')
        #        plt.grid()
        #        plt.savefig(f'../preproject/1d_heat_figures/{self.sol}_source_test_{t}.pdf')
        #        plt.show()

        #        # PNN
        #        plt.plot(self.disc.pts[self.disc.inner_ids1], pnnY[t], 'k',  label='exact temp')
        #        for model in self.pureNNs:
        #            plt.plot(self.disc.pts[self.disc.inner_ids1], model(np.array([pnnX[t]]))[0], 'r--',  label='pnn temp')
        #        plt.legend(title='trainset')
        #        plt.grid()
        #        plt.savefig(f'../preproject/1d_heat_figures/{self.sol}_temp_train_{t}.pdf')
        #        plt.show()

        #        plt.plot(self.disc.pts[self.disc.inner_ids1], pnnY_val[t], 'k',  label='exact temp')
        #        for model in self.pureNNs:
        #            plt.plot(self.disc.pts[self.disc.inner_ids1], model(np.array([pnnX_val[t]]))[0], 'r--',  label='pnn temp')
        #        plt.legend(title='valset')
        #        plt.grid()
        #        plt.savefig(f'../preproject/1d_heat_figures/{self.sol}_temp_val_{t}.pdf')
        #        plt.show()

        #        plt.plot(self.disc.pts[self.disc.inner_ids1], pnnY_test[t], 'k',  label='exact temp')
        #        for model in self.pureNNs:
        #            plt.plot(self.disc.pts[self.disc.inner_ids1], model(np.array([pnnX_test[t]]))[0], 'r--',  label='pnn temp')
        #        plt.legend(title='testset')
        #        plt.grid()
        #        plt.savefig(f'../preproject/1d_heat_figures/{self.sol}_temp_test_{t}.pdf')
        #        plt.show()


    def test(
            self, 
            interpol = True, 
            result_folder = None,
            ###intermediate_plot_interval = None
            ):
        '''
        interpol - (bool) use interpolating or extrapolating alpha set
        '''
        start_time = datetime.datetime.now()
        alphas = self.alpha_test_interpol if interpol else self.alpha_test_extrapol

        l2_devs={}
        L2_scores={}
        l2_scores={}
        ###intermediate_solutions={}
        final_solutions={}

        for i, alpha in enumerate(alphas):
            self.sol.set_alpha(alpha)

            # Define callback function to store l2 error development
            def relative_l2_callback(t, u_approx):
                inner_u = u_approx[self.disc.inner_ids2] if len(u_approx)==self.disc.Nv else u_approx[0,:]
                self.l2_development.append(np.sqrt(np.mean((inner_u-self.disc.format_u(self.sol.u(self.disc.pts[self.disc.inner_ids1],t)))**2)) / np.sqrt(np.mean(self.disc.format_u(self.sol.u(self.disc.pts[self.disc.inner_ids1],t)**2))))
                ###if abs(t-self.disk.k*intermediate_plot_interval*(len(intermediate_solutions)+1)) < 1e-10:
                    ###intermediate_solutions[intermediate_plot_interval*(len(intermediate_solutions)+1)]['temp'] = u


            # Solve with FEM
            self.l2_development = []
            fem_model = self.disc.make_model(self.sol.f, self.sol.u, w_ex=self.sol.w)
            fem_model.solve(self.disc.time_steps, T = self.disc.T, callback = relative_l2_callback)

            # prepare plotting
            prev_name = ''
            L2_scores[alpha] = {'FEM' : [fem_model.relative_L2()]}
            def rel_l2() : return np.sqrt(np.mean((fem_model.u_fem-self.disc.format_u(fem_model.u_ex(self.disc.pts,self.disc.T)))**2)) / np.sqrt(np.mean(self.disc.format_u(fem_model.u_ex(self.disc.pts,self.disc.T)**2)))
            l2_scores[alpha] = {'FEM' : [rel_l2()]}
            l2_devs[alpha]= {'FEM' : [self.l2_development]}
            final_solutions[alpha] = {'FEM' : [fem_model.u_fem.tolist()]}
            
            for model in self.models:
                self.l2_development = []
                fem_model.u_fem = model(self.sol.f,self.sol.u,alpha,callback=relative_l2_callback, w_ex=self.sol.w) # store in fem_model for easy use of relative_l2 and soltion functoins
                if prev_name != model.name:
                    prev_name = model.name
                    L2_scores[alpha][model.name] = []
                    l2_scores[alpha][model.name] = []
                    l2_devs[alpha][model.name] = []
                    final_solutions[alpha][model.name] = []
                L2_scores[alpha][model.name].append(fem_model.relative_L2())
                l2_scores[alpha][model.name].append(rel_l2())
                l2_devs[alpha][model.name].append(self.l2_development)
                final_solutions[alpha][model.name].append(fem_model.u_fem.tolist())

        if result_folder != None:
            # save final_solutions and l2 devs
            with open(result_folder+f'sol{self.sol.name}_final_solutions.json', "w") as f:
                f.write(json.dumps(final_solutions))
            with open(result_folder+f'sol{self.sol.name}_l2_devs.json', "w") as f:
                f.write(json.dumps(l2_devs))

        print(f'\nTime testing: {datetime.datetime.now()-start_time}')
        print(f'scores, (l2):',l2_scores)

        return L2_scores, l2_scores

    def plot_results(
            self, 
            result_folder,
            interpol = True, 
            figname=None, 
            statplot = 5, 
            ignore_models=[],
            legend=True,
            make_2d_graph=True
            ):
        '''
        interpol - (bool) use interpolating or extrapolating alpha set
        figname - (string or None) filename to save figure to. Not saved if None. .pdf if added to the figname before saving, and should not be included in figname.
        statplot - (bool or int) if plots should be of mean and variance (instead of every solution). If int, then statplot is set to len(models)>statplot
        ignore_models - (list of str) names of models not to include in plots. FEM always included.
        legend - (bool) if legend should be shown in plots
        make_2d_graph - (bool) going from final solution to graphs2d takes a long time, if this is already done, put this to false to skip that step by loading from file
        '''
        cmap = cm.coolwarm#CMRmap#cm.coolwarm
        
        # load stuff to plot
        with open(result_folder+f'sol{self.sol.name}_final_solutions.json') as f:
            final_solutions = json.load(f)
        with open(result_folder+f'sol{self.sol.name}_l2_devs.json') as f:
            l2_devs = json.load(f)

        alphas = self.alpha_test_interpol if interpol else self.alpha_test_extrapol

        # prepare plotting
        figsize = (6,3.4)
        if type(statplot) == int:
            statplot = len(self.models) > statplot
        fig, axs = plt.subplots(1,len(alphas), figsize=figsize)

        graphs1d={}
        graphs2d={}
        if self.disc.dim==2:
            N = len(self.disc.pts_line)-1
            X2d = np.array([np.linspace(self.disc.xa,self.disc.xb,N+1)]*(N+1)).T
            Y2d = np.array([np.linspace(self.disc.ya,self.disc.yb,N+1)]*(N+1))
            pts2d = np.array([X2d,Y2d]).T

        for i, alpha in enumerate(alphas):
            self.sol.set_alpha(alpha)

            fem_model = self.disc.make_model(self.sol.f, self.sol.u, w_ex=self.sol.w)
            graphs1d[alpha]={}
            graphs2d[alpha] = {}
            prev_name = ''
            j=0

            for model in self.models:
                if not model.name in ignore_models:
                    self.l2_development = []
                    if prev_name != model.name:
                        prev_name = model.name
                        j=0
                        graphs1d[alpha][model.name] = []

                        if self.disc.dim==2:
                            graphs2d[alpha][model.name] = []
                    fem_model.u_fem = np.array(final_solutions[f'{alpha}'][model.name][j])
                    if self.disc.dim==2 and make_2d_graph:
                        graphs2d[alpha][model.name].append(fem_model.solution(pts2d).tolist())
                    if self.disc.equation == 'elasticity':
                        graphs1d[alpha][model.name].append(fem_model.solution(self.disc.pts_line)[:,0])
                    else:
                        graphs1d[alpha][model.name].append(fem_model.solution(self.disc.pts_line))
                    j+=1
            fem_model.u_fem = np.array(final_solutions[f'{alpha}']['FEM'][0])
            graphs1d[alpha]['FEM'] = [fem_model.solution(self.disc.pts_line)]
            graphs1d[alpha]['exact'] = [self.sol.u(self.disc.pts_line)]
            if self.disc.dim==2:
                if self.disc.equation == 'elasticity':
                    graphs1d[alpha]['FEM'] = [fem_model.solution(self.disc.pts_line)[:,0]]
                    graphs1d[alpha]['exact'] = [self.sol.u(self.disc.pts_line)[:,0]] # only first component is plotted in graph1d for elasticity
                graphs2d[alpha]['FEM'] = fem_model.solution(pts2d).tolist()
                graphs2d[alpha]['exact'] = self.sol.u(pts2d).tolist()

        # save graphs2d 
        if self.disc.dim==2:
            if make_2d_graph:
                with open(result_folder+f'sol{self.sol.name}_graphs2d.json', "w") as f:
                    f.write(json.dumps(graphs2d))
            with open(result_folder+f'sol{self.sol.name}_graphs2d.json') as f:
                graphs2d = json.load(f)

        # plot 1d final solutions
        for i, alpha in enumerate(alphas):
            self.sol.set_alpha(alpha)
            for name in graphs1d[alpha]:
                curr_graphs = np.array(graphs1d[alpha][name])
                if statplot and not name in ['FEM', 'exact', 'FEM_2']:
                    mean = np.mean(curr_graphs, axis=0)
                    std = np.std(curr_graphs, axis=0, ddof=1) # reduce one degree of freedom due to mean calculation
                    if self.disc.dim==1:
                        axs[i].plot(self.disc.pts_line, mean, color=COLORS[name])
                        axs[i].fill_between(self.disc.pts_line, mean+std, mean-std, color=COLORS[name], alpha = 0.4, label = name)
                    if self.disc.dim==2:
                        axs[i].plot(self.disc.pts_line[:,0], mean, color=COLORS[name])
                        axs[i].fill_between(self.disc.pts_line[:,0], mean+std, mean-std, color=COLORS[name], alpha = 0.4, label = name)
                else:
                    line=self.disc.pts_line
                    if self.disc.dim==2:
                        line=line[:,0]
                    for k, graph in enumerate(curr_graphs):
                        axs[i].plot(line, graph, color=COLORS[name], label=name if k==0 else None, linestyle = '--' if name=='exact' else '-')
            axs[i].grid()
            if legend:
                axs[i].legend(title=f'sol={self.sol.name},a={alpha}')

        plt.tight_layout()
        if figname != None:
            plt.savefig(figname+'.pdf')
        if self.plot:# and self.disc.dim==1:
            plt.show()
        else:
            plt.close()

        # Plot l2 development:
        fig, axs = plt.subplots(1,len(alphas), figsize=figsize)
        for i, alpha in enumerate(alphas):
            axs[i].set_yscale('log')
            for name in l2_devs[f'{alpha}']:
                if not name in ignore_models:
                    curr_devs = np.array(l2_devs[f'{alpha}'][name])
                    mean = np.mean(curr_devs, axis=0)
                    if statplot and not name in ['FEM', 'FEM_2']:
                        axs[i].plot(np.arange(len(mean)), mean, color=COLORS[name])
                        std = np.std(curr_devs, axis=0, ddof=1) # reduce one degree of freedom due to mean calculation
                        axs[i].fill_between(np.arange(len(mean)), mean+std, mean, color=COLORS[name], alpha = 0.4, label = name)
                    else:
                        for k, dev in enumerate(curr_devs):
                            axs[i].plot(np.arange(len(mean)), dev, color=COLORS[name], label=name if k==0 else None)

            axs[i].grid()
            if legend:
                axs[i].legend(title=f'sol={self.sol.name},a={alpha}')
            if axs[i].get_ylim()[1] > 1e3: # if error divergres (>1e3), it plot is capped at 1e2 to better view the interesting stuff
                axs[i].set_ylim(top=1e2, bottom=axs[i].get_ylim()[0]*8) # bottom is also increased as a workaround to decrease the large white space at the bottom.

        plt.tight_layout()
        if figname != None:
            plt.savefig(figname+'_devs.pdf')
        if self.plot:
            plt.show()
        else:
            plt.close()

        # plot 2d stuff
        def add_cbar(im, ax, ud):
            ax.invert_yaxis()
            ax.set_xticks([0,len(self.disc.pts_line)])
            ax.set_xticklabels([0,1]) # NOTE: hardcoded x,y \in [0,1]
            ax.set_yticks([0,len(self.disc.pts_line)])
            ax.set_yticklabels([0,1])
            minval, maxval = im.get_clim()
            minval = min(minval, -maxval)
            maxval = max(maxval, -minval)
            im.set_clim(minval, maxval)
            cbar = plt.colorbar(im, ax=ax, location='bottom' if ud==2 else None)
            cbar.ax.tick_params(rotation=45)
        if self.disc.dim==2 and statplot:
            ud = self.disc.udim
            nw = ud * len(alphas)
            nh = len(self.modelnames)+2 # + models + FEM and exact
            fig = plt.figure(figsize=(4.2*len(alphas)*ud**0.5,3.2*nh*ud**0.3))
            for i, alpha in enumerate(alphas):
                for j in range(ud):
                    if ud==1:
                        u_ex=np.array(graphs2d[f'{alpha}']['exact'])
                        u_fem=np.array(graphs2d[f'{alpha}']['FEM'])
                    if ud==2:
                        u_ex=np.array(graphs2d[f'{alpha}']['exact'])[:,:,j]
                        u_fem=np.array(graphs2d[f'{alpha}']['FEM'])[:,:,j]
                    ax = fig.add_subplot(nh,nw,ud*i+j+1)
                    im = ax.imshow(u_ex, cmap=cmap)
                    ax.set_xticks([0,len(self.disc.pts_line)])
                    ax.set_xticklabels([0,1])
                    ax.set_yticks([0,len(self.disc.pts_line)])
                    ax.set_yticklabels([0,1])
                    ax.invert_yaxis()
                    cbar = plt.colorbar(im, ax=ax, location='bottom' if ud==2 else None)
                    cbar.ax.tick_params(rotation=45)
                    plt.title(fr'exact u[{j}],$\alpha$={alpha}')
                    ax = fig.add_subplot(nh,nw,ud*i+j+nw+1)
                    im = ax.imshow(u_fem-u_ex, cmap=cmap)
                    add_cbar(im, ax, ud)
                    plt.title(f'PBM error')
                    for k, name in enumerate(self.modelnames):
                        if not model.name in ignore_models:
                            if ud==1:
                                u_mean = np.mean(np.array(graphs2d[f'{alpha}'][name]),axis=0)
                            if ud==2:
                                u_mean = np.mean(np.array(graphs2d[f'{alpha}'][name]),axis=0)[:,:,j]
                            ax = fig.add_subplot(nh,nw,ud*i+j+nw*(2+k)+1)
                            im = ax.imshow(u_mean-u_ex, cmap=cmap)
                            add_cbar(im, ax, ud)
                            plt.title('mean '+{"CoSTA_DNN":"CoSTA", "DNN":"DDM", "FEM":"PBM"}[name]+' error')

            plt.tight_layout()
            if figname != None:
                plt.savefig(figname+'_2d.pdf')
            if self.plot:
                plt.show()
            else:
                plt.close()

            nh = len(self.modelnames)+1 # + models + FEM and exact
            fig = plt.figure(figsize=(4.2*nw,3.2*nh))
            for i, alpha in enumerate(alphas):
                for j in range(ud):
                    if ud==1:
                        u_ex=np.array(graphs2d[f'{alpha}']['exact'])
                    if ud==2:
                        u_ex=np.array(graphs2d[f'{alpha}']['exact'])[:,:,j]
                    ax = fig.add_subplot(nh,nw,ud*i+j+1)
                    im = ax.imshow(u_ex, cmap=cmap)
                    plt.colorbar(im, ax=ax, location='bottom' if ud==2 else None)
                    plt.title(fr'exact u[{j}],$\alpha$={alpha}')
                    for k, name in enumerate(self.modelnames):
                        if not model.name in ignore_models:
                            if ud==1:
                                u_std = np.std(np.array(graphs2d[f'{alpha}'][name]),axis=0, ddof=1)
                            if ud==2:
                                u_std = np.std(np.array(graphs2d[f'{alpha}'][name]),axis=0, ddof=1)[:,:,j]
                            ax = fig.add_subplot(nh,nw,ud*i+j+nw*(1+k)+1)
                            im = ax.imshow(u_std, cmap=cmap)
                            add_cbar(im, ax, ud)
                            plt.title('mean '+{"CoSTA_DNN":"CoSTA", "DNN":"DDM", "FEM":"PBM"}[name]+' error')

            plt.tight_layout()
            if figname != None:
                plt.savefig(figname+'_2d_std.pdf')
            if self.plot:
                plt.show()
            else:
                plt.close()
