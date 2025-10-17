import numpy as np
import quadrature
import FEM
import solvers
import functions
import parameters
from matplotlib import pyplot as plt
import sys
import utils

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

Ne, time_steps, DNNkwargs, pgDNNkwargs, LSTMkwargs, pgLSTMkwargs, NoM, time_delta = parameters.set_args(mode = mode, dim=1)
pgLRkwargs= {'lr':5e-3,'patience':[10,20], 'epochs':[100,100], 'min_epochs':[50,50], 'bn_depth':pgDNNkwargs['bn_depth']}#, 'l1_penalty':0.01}

if len(sys.argv)>2:
    if sys.argv[2]=='f':
        source = True 
    elif sys.argv[2]=='0':
        source = False
    else:
        print('failed reading source term')
        source = False
else:
    source = False
print('Using exact source' if source else 'Using unknown source (i.e. guessing zero)')

if len(sys.argv)>3:
    p = int(sys.argv[3])
else:
    p=1
print(f'Using p={p}')

modelnames = {
        'DNN' : NoM,
        'pgDNN' : NoM,
        'pgLR' : NoM,
        #'LSTM' : NoM,
        'CoSTA_DNN' : NoM,
        'CoSTA_pgDNN' : NoM,
        'CoSTA_pgLR' : NoM,
        #'CoSTA_LSTM' : NoM,
        }
NNkwargs = {
        'DNN':DNNkwargs, 
        'CoSTA_DNN':DNNkwargs,
        'pgDNN' : pgDNNkwargs,
        'pgLR' : pgLRkwargs,
        'CoSTA_pgDNN':pgDNNkwargs,
        'CoSTA_pgLR':pgLRkwargs,
        'LSTM':LSTMkwargs, 
        'CoSTA_LSTM':LSTMkwargs, 
        }

for sol_index in [4,1,2,3]:
    print(f'sol_index: {sol_index}\n')
    f,u = functions.SBMFACT[sol_index]
    extra_tag = '_LR1' # for different names when testing specific stuff
    pgDNNkwargs['activation'] = None
    sol = functions.Solution(T=5, f_raw=f, u_raw=u, zero_source=not source, name=f'{sol_index}', time_delta=time_delta)
    model = solvers.Solvers(modelnames=modelnames, p=p,sol=sol, Ne=Ne, time_steps=time_steps, NNkwargs=NNkwargs)
    figname = f'../preproject/1d_heat_figures/{mode}/{"known_f" if source else "unknown_f"}/interpol/loss_sol{sol_index}_p{p}{extra_tag}.pdf'
    model_folder = f'../preproject/saved_models/{"known_f" if source else "unknown_f"}/{mode}{extra_tag}/'#_explosions/'
    #figname = None
    model.plot=False
    model.train(figname=figname, model_folder = model_folder)
    #model.train(figname=figname)
    #model.load_weights(model_folder)

    #model.plot=True

    # Interpolation
    result_folder = f'../preproject/saved_results/{"known_f" if source else "unknown_f"}/{mode}{extra_tag}/interpol/'
    utils.makefolder(result_folder)
    _ = model.test(interpol = True, result_folder=result_folder)
    figname = f'../preproject/1d_heat_figures/{mode}/{"known_f" if source else "unknown_f"}/interpol/sol{sol_index}_p{p}{extra_tag}'
    model.plot_results(result_folder=result_folder, interpol = True, figname=figname, statplot = 5)
    figname = f'../preproject/1d_heat_figures/{mode}/{"known_f" if source else "unknown_f"}/interpol/sol{sol_index}_p{p}_nonstat{extra_tag}'
    model.plot_results(result_folder=result_folder, interpol = True, figname=figname, statplot = False)

    # Extrapolation
    result_folder = f'../preproject/saved_results/{"known_f" if source else "unknown_f"}/{mode}{extra_tag}/extrapol/'
    utils.makefolder(result_folder)
    _ = model.test(interpol = False, result_folder=result_folder)
    figname = f'../preproject/1d_heat_figures/{mode}/{"known_f" if source else "unknown_f"}/extrapol/sol{sol_index}_p{p}{extra_tag}'
    model.plot_results(result_folder=result_folder, interpol = False, figname=figname, statplot = 5)
    figname = f'../preproject/1d_heat_figures/{mode}/{"known_f" if source else "unknown_f"}/extrapol/sol{sol_index}_p{p}_nonstat{extra_tag}'
    model.plot_results(result_folder=result_folder, interpol = False, figname=figname, statplot = False)

    #extra_tag = '_xxpol'
    #model.alpha_test_extrapol = [-0.9,4]
    #_ = model.test(interpol = False, figname=figname, ignore_models=['pgDNN'])
