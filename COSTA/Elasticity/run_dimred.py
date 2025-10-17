import numpy as np
import quadrature
import FEM
import solvers
import functions
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt, cm
import sys
import sympy as sp
import utils
import parameters



mode = 'bugfix'
if len(sys.argv)>1:
    mode = {'0':'bugfix',
            '1':'quick_test',
            '2':'full_test'}[sys.argv[1]]
else:
    print('No mode specified')
    print('add arg for mode:')
    print('0: bufix (default)')
    print('1: quick_test')
    print('2: full_test')
    print('syntax e.g.: python testing.py 2')

Ne, time_steps, DNNkwargs, pgDNNkwargs, LSTMkwargs, pgLSTMkwargs, NoM, time_delta = parameters.set_args(mode = mode, dim=2)

if len(sys.argv)>2:
    p = int(sys.argv[2])
else:
    p=1
print(f'Using p={p}')

modelnames = {
        'DNN' : NoM,
        #'pgDNN' : NoM,
        #'LSTM' : NoM,
        'CoSTA_DNN' : NoM,
        #'CoSTA_pgDNN' : NoM,
        #'CoSTA_LSTM' : NoM,
        }
NNkwargs = {
        'DNN':DNNkwargs,
        'CoSTA_DNN':DNNkwargs,
        'pgDNN' : pgDNNkwargs,
        'CoSTA_pgDNN':pgDNNkwargs,
        'LSTM':LSTMkwargs,
        'CoSTA_LSTM':LSTMkwargs,
        }


xa,xb,ya,yb = 0,1,0,1
for i in [0 ,4, 5, 1]:
    print(f'sol_index: {i}\n')
    f,u = functions.dimred[i]
    sol = functions.Solution(T=1, f_raw=f, u_raw=u, zero_source=False, name=f'DR_{i}', time_delta=time_delta)
    
    model = solvers.Solvers(modelnames=modelnames, p=p,sol=sol, Ne=Ne, time_steps=time_steps,xa=xa, xb=xb, ya=ya,yb=yb,dim=2, skip_create_data=True, NNkwargs=NNkwargs)
    extra_tag = '' # for different names when testing specific stuff
    figname = f'../master/bp_heat_figures/{mode}/interpol/loss_sol{i}{extra_tag}'
    model_folder = f'../master/saved_models/bp_heat/{mode}{extra_tag}/interpol/'
    model.plot=False
    #model.train(figname=figname, model_folder = model_folder)
    #model.load_weights(model_folder)
    
    #ignore_models = ['pgDNN', 'CoSTA_pgDNN']
    ignore_models = []
    legend=False
    make_2d_graph=False
    
    # Interpolation
    #result_folder = f'../master/saved_results/bp_heat/{mode}{extra_tag}/interpol/'
    #utils.makefolder(result_folder)
    ##_ = model.test(interpol = True, result_folder=result_folder)
    figname = f'../master/bp_heat_figures/{mode}/interpol/sol{i}{extra_tag}'
    model.plot_results(result_folder=result_folder, interpol = True, figname=figname, statplot = 2, ignore_models = ignore_models, legend=legend, make_2d_graph=make_2d_graph)
    #figname = f'../master/bp_heat_figures/{mode}/interpol/sol{x}_nonstat{extra_tag}'
    #model.plot_results(result_folder=result_folder, interpol = True, figname=figname, statplot = False)
    
    # Extrapolation
    result_folder = f'../master/saved_results/bp_heat/{mode}{extra_tag}/extrapol/'
    utils.makefolder(result_folder)
    #_ = model.test(interpol = False, result_folder=result_folder)
    figname = f'../master/bp_heat_figures/{mode}/extrapol/sol{i}{extra_tag}'
    model.plot_results(result_folder=result_folder, interpol = False, figname=figname, statplot = 5, ignore_models = ignore_models, legend=legend, make_2d_graph=make_2d_graph)
    #figname = f'../master/bp_heat_figures/{mode}/extrapol/sol{x}_nonstat{extra_tag}'
    #model.plot_results(result_folder=result_folder, interpol = False, figname=figname, statplot = False)
