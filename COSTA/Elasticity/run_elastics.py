import numpy as np
import quadrature
import FEM
import solvers
import functions
from matplotlib import pyplot as plt, cm
import sys
import sympy as sp
import utils
import parameters

mode = 'bugfix'
if len(sys.argv)>1:
    mode = {'0':'bugfix',
            '1':'quick_test',
            '2':'full_test',
            '3':'el_test'}[sys.argv[1]]
else:
    print('No mode specified')
    print('add arg for mode:')
    print('0: bufix (default)')
    print('1: quick_test')
    print('2: full_test')
    print('syntax e.g.: python testing.py 2')

Ne, time_steps, DNNkwargs, pgDNNkwargs, LSTMkwargs, pgLSTMkwargs, NoM, time_delta = parameters.set_args(mode = mode, dim=2)
#time_steps = 100

if len(sys.argv)>2:
    if sys.argv[2]=='bp':
        source = 'reduced_source'
    elif sys.argv[2]=='f':
        source = 'exact_source'
    elif sys.argv[2]=='0':
        source = 'zero_source'
    elif sys.argv[2]=='nl':
        source = 'non_linear'
    else:
        print('failed reading source term')
        source = 'exact_source'
else:
    source = 'exact_source'
print('Using', source)

if len(sys.argv)>3:
    static = bool(sys.argv[3])
else:
    static = False
print(f'Using {"STATIC" if static else "TRANSIENT"} version of elastic equation')

p=1

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
        'pgLSTM':pgLSTMkwargs,
        'CoSTA_LSTM':LSTMkwargs,
        'CoSTA_pgLSTM':pgLSTMkwargs,
        }

#assert not (static and ('LSTM' in modelnames or 'CoSTA_LSTM' in modelnames))

xa,xb,ya,yb = 0,1,0,1
for i in [0,1,2]:
    print(f'sol_index: {i}\n')
    T = 1
    if source == 'reduced_source':
        f,u,w = functions.manufacture_elasticity_solution(d1=3, d2=2, static=static, **functions.ELsols3d[i])
    else:
        f,u,w = functions.manufacture_elasticity_solution(d1=2, d2=2, static=static, non_linear=source=='non_linear', **functions.ELsols[i])
    sol = functions.Solution(T=T, f_raw=f, u_raw=u, zero_source=source=='zero_source', name=f'ELsol{i}',w_raw=w)
    
    model = solvers.Solvers(equation='elasticity', static=static, modelnames=modelnames, p=p,sol=sol, Ne=Ne, time_steps=time_steps,xa=xa, xb=xb, ya=ya,yb=yb,dim=2,skip_create_data=True, NNkwargs=NNkwargs)
    extra_tag = '' # for different names when testing specific stuff
    figname = f'../master/2d_elastic_figures/{source}/{mode}/{"static_"if static else ""}interpol/loss_sol{i}{extra_tag}'
    model_folder = f'../master/saved_models/2d_elastic/{"static_"if static else ""}{source}/{mode}{extra_tag}/'#_explosions/'
    model.plot=False
    #model.train(figname=figname, model_folder = model_folder)
    #model.load_weights(model_folder)
    
    legend=False
    make_2d_graph=False

    # Interpolation
    result_folder = f'../master/saved_results/2d_elastic/{"static_"if static else ""}{source}/{mode}{extra_tag}/interpol/'
    utils.makefolder(result_folder)
    #_ = model.test(interpol = True, result_folder=result_folder)
    figname = f'../master/2d_elastic_figures/{source}/{mode}/{"static_"if static else ""}interpol/sol{i}{extra_tag}'
    model.plot_results(result_folder=result_folder, interpol = True, figname=figname, statplot = 5, legend=legend, make_2d_graph=make_2d_graph)
    #figname = f'../master/2d_elastic_figures/{source}/{mode}/{"static_"if static else ""}interpol/sol{i}_nonstat{extra_tag}'
    #model.plot_results(result_folder=result_folder, interpol = True, figname=figname, statplot = False)

    # Extrapolation
    result_folder = f'../master/saved_results/2d_elastic/{"static_"if static else ""}{source}/{mode}{extra_tag}/extrapol/'
    utils.makefolder(result_folder)
    #_ = model.test(interpol = False, result_folder=result_folder)
    figname = f'../master/2d_elastic_figures/{source}/{mode}/{"static_"if static else ""}extrapol/sol{i}{extra_tag}'
    model.plot_results(result_folder=result_folder, interpol = False, figname=figname, statplot = 5, legend=legend, make_2d_graph=make_2d_graph)
    #figname = f'../master/2d_elastic_figures/{source}/{mode}/{"static_"if static else ""}extrapol/sol{i}_nonstat{extra_tag}'
    #model.plot_results(result_folder=result_folder, interpol = False, figname=figname, statplot = False)
