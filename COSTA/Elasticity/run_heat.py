import numpy as np
import quadrature
import FEM
import solvers
import functions
from matplotlib import pyplot as plt
import sys
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

Ne, time_steps, DNNkwargs, pgDNNkwargs, LSTMkwargs, pgLSTMkwargs, NoM, time_delta = parameters.set_args(mode = mode, dim=1)

if len(sys.argv)>2:
    if sys.argv[2]=='f':
        source = True 
    elif sys.argv[2]=='0':
        source = False
    else:
        print('failed reading source term')
        source = True
else:
    source = True
print('Using exact source' if source else 'Using unknown source (i.e. guessing zero)')

if len(sys.argv)>3:
    p = int(sys.argv[3])
else:
    p=1
print(f'Using p={p}')
modelnames = {
        'DNN' : NoM,
        #'pgDNN' : NoM,
        #'LSTM' : NoM,
        #'pgLSTM' : NoM,
        'CoSTA_DNN' : NoM,
        #'CoSTA_pgDNN' : NoM,
        #'CoSTA_LSTM' : NoM,
        #'CoSTA_pgLSTM' : NoM,
        }
NNkwargs = {
        'DNN':DNNkwargs, 
        'CoSTA_DNN':DNNkwargs,
        'pgDNN' : pgDNNkwargs,
        'CoSTA_pgDNN':pgDNNkwargs,
        'LSTM':LSTMkwargs, 
        'CoSTA_LSTM':LSTMkwargs, 
        'pgLSTM':pgLSTMkwargs, 
        'CoSTA_pgLSTM':pgLSTMkwargs, 
        }

for sol_index in [5,6]:
    print(f'sol_index: {sol_index}\n')
    f,u = functions.var_k[sol_index]
    solname = f'var_k{sol_index}'
    T = 1
    sol = functions.Solution(T=T, f_raw=f, u_raw=u, zero_source=not source, name=solname, time_delta=time_delta)
    model = solvers.Solvers(modelnames=modelnames, p=p,sol=sol, Ne=Ne, time_steps=time_steps, skip_create_data=False, NNkwargs=NNkwargs)
    extra_tag = '' # for different names when testing specific stuff
    figname = f'../master/1d_heat_figures/{"known_f" if source else "unknown_f"}/{mode}/interpol/loss_sol{sol_index}{extra_tag}'
    model.plot=False
    model_folder = f'../master/saved_models/1d_heat/{"known_f" if source else "unknown_f"}/{mode}{extra_tag}/'
    model.train(figname=figname, model_folder = model_folder)
    #model.load_weights(model_folder)

    #ignore_models = ['pgDNN', 'CoSTA_pgDNN']
    ignore_models = []
    legend=False

    # Interpolation
    midplots = 1
    for i in range(midplots):
        model.disc.T = (i+1)*T/midplots
        model.sol.T = model.disc.T
        model.disc.time_steps = ((i+1)*time_steps)//midplots
        extra_tag_2 = f'{extra_tag}_t{model.disc.time_steps}' if i<midplots-1 else extra_tag

        result_folder = f'../master/saved_results/1d_heat/{"known_f" if source else "unknown_f"}/{mode}{extra_tag_2}/interpol/'
        utils.makefolder(result_folder)
        _ = model.test(interpol = True, result_folder=result_folder)
        figname = f'../master/1d_heat_figures/{"known_f" if source else "unknown_f"}/{mode}/interpol/sol{sol_index}{extra_tag_2}'
        model.plot_results(result_folder=result_folder, interpol = True, figname=figname, statplot = 4, ignore_models = ignore_models, legend=legend)
        #figname = f'../master/1d_heat_figures/{"known_f" if source else "unknown_f"}/{mode}/interpol/sol{sol_index}_nonstat{extra_tag_2}'
        #model.plot_results(result_folder=result_folder, interpol = True, figname=figname, statplot = False, ignore_models = ignore_models, legend=legend)

    # Extrapolation
    for i in range(midplots):
        model.disc.T = (i+1)*T/midplots
        model.sol.T = model.disc.T
        model.disc.time_steps = ((i+1)*time_steps)//midplots
        extra_tag_2 = f'{extra_tag}_t{model.disc.time_steps}' if i<midplots-1 else extra_tag

        result_folder = f'../master/saved_results/1d_heat/{"known_f" if source else "unknown_f"}/{mode}{extra_tag_2}/extrapol/'
        utils.makefolder(result_folder)
        _ = model.test(interpol = False, result_folder=result_folder)
        figname = f'../master/1d_heat_figures/{"known_f" if source else "unknown_f"}/{mode}/extrapol/sol{sol_index}{extra_tag_2}'
        model.plot_results(result_folder=result_folder, interpol = False, figname=figname, statplot = 4, ignore_models = ignore_models, legend=legend)
        #figname = f'../master/1d_heat_figures/{"known_f" if source else "unknown_f"}/{mode}/extrapol/sol{sol_index}_nonstat{extra_tag_2}'
        #model.plot_results(result_folder=result_folder, interpol = False, figname=figname, statplot = False, ignore_models = ignore_models, legend=legend)
